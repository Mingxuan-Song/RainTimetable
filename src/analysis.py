from pathlib import Path
import itertools
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

COLORS = ['#237d73', '#dc8644', '#5478a4', '#a46791', '#5c685e', '#b89b24']
plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.prop_cycle': plt.cycler(color=COLORS)})

def savefig(root, name):
    path = Path(root) / 'outputs/figures'
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / f'{name}.png', bbox_inches='tight')
    from IPython import get_ipython
    if get_ipython() is not None:
        from IPython.display import display, Image
        display(Image(filename=str(path / f'{name}.png')))
    plt.close()

def table(root, name, frame):
    path = Path(root) / 'outputs/tables'
    path.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path / f'{name}.csv', index=False)
    return frame

def read_frame(path, required):
    frame = pd.read_csv(path)
    if not set(['date'] + required).issubset(frame.columns):
        raise ValueError(f'Missing columns: {path.name}')
    frame['date'] = pd.to_datetime(frame['date'], format='%Y-%m-%d', errors='raise')
    if frame['date'].duplicated().any():
        raise ValueError(f'Duplicate dates: {path.name}')
    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors='raise')
        if np.isinf(frame[col]).any():
            raise ValueError(f'Infinite values: {col}')
    return frame.set_index('date').sort_index()[required]

def load_validate(root):
    raw = Path(root) / 'data/raw'
    rain = read_frame(raw / 'GreatPlains_CHIRPS_DailyPrecip_2000_2024.csv', ['precip_daily'])
    veg = read_frame(raw / 'GreatPlains_MOD09A1_NDVI_8day_2000_2024.csv', ['ndvi'])
    weather = read_frame(raw / 'GreatPlains_ERA5L_SoilTempPET_Daily_2000_2024.csv',
                         ['temp2m_daily_K', 'soil_moisture_daily', 'pet_daily_m'])
    if (rain.precip_daily.dropna() < 0).any():
        raise ValueError('Negative precipitation')
    if not veg.ndvi.dropna().between(-1, 1).all():
        raise ValueError('NDVI outside physical range')
    if not weather.temp2m_daily_K.dropna().between(180, 350).all():
        raise ValueError('Implausible temperature')
    if not weather.soil_moisture_daily.dropna().between(0, 1).all():
        raise ValueError('Implausible volumetric soil moisture')
    for name, frame in [('rain', rain), ('weather', weather)]:
        if not frame.index.equals(pd.date_range(frame.index.min(), frame.index.max(), freq='D')):
            raise ValueError(f'Gaps in daily {name} dates')
        if frame.isna().any().any():
            raise ValueError(f'Missing daily {name} values')
    # MODIS composites restart on January 1, so the last interval can be 5/6 days.
    expected_veg = pd.DatetimeIndex([pd.Timestamp(y, 1, 1) + pd.Timedelta(days=k)
           for y in range(veg.index.min().year, veg.index.max().year + 1)
           for k in range(0, 365 + int(pd.Timestamp(y, 12, 31).is_leap_year), 8)
           if veg.index.min() <= pd.Timestamp(y, 1, 1) + pd.Timedelta(days=k) <= veg.index.max()])
    if len(veg.index.difference(expected_veg)):
        raise ValueError('Unexpected composite dates')
    audit = pd.DataFrame([dict(source=name, rows=len(frame), start=str(frame.index.min().date()),
                    end=str(frame.index.max().date()), missing_values=int(frame.isna().sum().sum()),
                    duplicate_dates=0, absent_dates=int(len(expected_veg.difference(frame.index))) if name == 'MODIS NDVI' else 0) for name, frame in [('CHIRPS', rain), ('MODIS NDVI', veg), ('ERA5-Land', weather)]])
    table(root, 'data_audit', audit)
    daily = rain.join(weather, how='inner')
    daily['temp_c'] = daily['temp2m_daily_K'] - 273.15
    complete = []
    for year, group in daily.groupby(daily.index.year):
        expected = 366 if pd.Timestamp(year, 12, 31).is_leap_year else 365
        if len(group) == expected and year >= 2001:
            complete.append(year)
    daily = daily.loc[daily.index.year.isin(complete)].copy()
    veg = veg.loc[veg.index.year.isin(complete)].copy()
    # Missing NDVI targets are omitted; no interpolation supplies target values.
    good_veg = veg.dropna().copy()
    for name, frame in [('daily', daily), ('vegetation', good_veg)]:
        frame.to_csv(Path(root) / 'data/processed' / f'{name}.csv')
    return daily, good_veg, audit

def longest_run(mask):
    """Return the length and inclusive positions of the longest True run."""
    best, start, end, current, current_start = 0, -1, -1, 0, 0
    for i, value in enumerate(mask):
        if value:
            if current == 0:
                current_start = i
            current += 1
            if current > best:
                best, start, end = current, current_start, i
        else:
            current = 0
    return best, start, end

def rhythm(values, threshold=1.0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Rainfall window must contain finite, nonnegative values')
    total = values.sum()
    shares = values / total if total > 0 else np.zeros(len(values))
    k = max(1, int(np.ceil(len(values) * .10)))
    return dict(rain_total=float(total), wet_days=int((values >= threshold).sum()),
                dry_run=int(longest_run(values < threshold)[0]),
                concentration=float((shares ** 2).sum()),
                top10_share=float(np.sort(values)[-k:].sum() / total) if total > 0 else 0.)

def annual_summary(root, daily, veg, threshold=1.):
    records = []
    for year, group in daily.groupby(daily.index.year):
        season = group.loc[group.index.month.isin(range(4, 10))]
        v = veg.loc[(veg.index.year == year) & veg.index.month.isin(range(4, 10)), 'ndvi']
        if len(v) < 20:
            continue
        r = rhythm(season.precip_daily, threshold)
        records.append(dict(year=year, annual_rain=group.precip_daily.sum(),
                            season_temp=season.temp_c.mean(), ndvi_mean=v.mean(),
                            ndvi_min=v.min(), ndvi_peak=v.max(), ndvi_count=len(v), **r))
    return table(root, 'annual_summary', pd.DataFrame(records))

def match_years(annual, tolerance=.10):
    """Enumerate all pairs with similar annual totals, without using NDVI."""
    records = []
    for i, j in itertools.combinations(range(len(annual)), 2):
        a, b = annual.iloc[i], annual.iloc[j]
        gap = abs(a.annual_rain - b.annual_rain) / ((a.annual_rain + b.annual_rain) / 2)
        if gap <= tolerance:
            # Orient pairs from smaller to larger seasonal rain concentration.
            low, high = (a, b) if a.concentration <= b.concentration else (b, a)
            records.append(dict(year_low=int(low.year), year_high=int(high.year),
                                rain_gap=gap, concentration_difference=high.concentration-low.concentration,
                                ndvi_difference=high.ndvi_mean-low.ndvi_mean,
                                dry_run_difference=high.dry_run-low.dry_run,
                                seasonal_rain_difference=high.rain_total-low.rain_total))
    return pd.DataFrame(records)

def plot_overview(root, daily, veg):
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6))
    axes[0].plot(veg.index, veg.ndvi, lw=.9)
    axes[0].set(ylabel='Regional NDVI', title='Vegetation observations retained for complete years')
    m = daily.precip_daily.resample('MS').sum()
    axes[1].bar(m.index, m.values, width=23, color=COLORS[1])
    axes[1].set(ylabel='Monthly rain (mm)', xlabel='Year')
    fig.tight_layout(); savefig(root, '01_overview')

def plot_seasonal(root, daily, veg):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    vm = veg.assign(month=veg.index.month).groupby('month').ndvi
    q = vm.quantile([.1, .5, .9]).unstack()
    axes[0].plot(q.index, q[.5], marker='o');axes[0].fill_between(q.index, q[.1], q[.9], alpha=.18)
    axes[0].set(xlabel='Month', ylabel='NDVI', title='Median and 10th--90th percentiles')
    rm = daily.precip_daily.resample('MS').sum()
    q2 = rm.groupby(rm.index.month).quantile([.1, .5, .9]).unstack()
    axes[1].plot(q2.index, q2[.5], marker='o', color=COLORS[1]);axes[1].fill_between(q2.index, q2[.1], q2[.9], alpha=.18, color=COLORS[1])
    axes[1].set(xlabel='Month', ylabel='Monthly rain (mm)', title='Rainfall seasonality across years')
    for ax in axes: ax.set_xticks(range(1, 13))
    fig.tight_layout();savefig(root, '02_seasonality')

def plot_annual(root, annual, pairs):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5))
    sc=axes[0].scatter(annual.annual_rain, annual.ndvi_mean, c=annual.dry_run, cmap='viridis', s=45)
    axes[0].set(xlabel='Annual rain (mm)', ylabel='April--September mean NDVI', title='Complete years')
    fig.colorbar(sc, ax=axes[0], label='Longest low-rain run (days)')
    axes[1].scatter(pairs.concentration_difference, pairs.ndvi_difference, alpha=.7)
    axes[1].axhline(0, color='gray', lw=.8)
    axes[1].set(xlabel='Difference in rain concentration', ylabel='Difference in seasonal NDVI', title='All pairs within 10% annual rainfall')
    fig.tight_layout();savefig(root, '03_matched_pairs')

def choose_pair(pairs):
    # The illustration is selected on rainfall alone, before examining NDVI.
    return pairs.sort_values(['concentration_difference', 'year_low', 'year_high'], ascending=[False, True, True]).iloc[0]

def plot_pair(root, daily, veg, pair):
    years=[int(pair.year_low), int(pair.year_high)]
    fig, axes=plt.subplots(2, 2, figsize=(9.5, 5.2), sharex='col', sharey='row')
    for j, year in enumerate(years):
        d=daily.loc[(daily.index.year==year)&daily.index.month.isin(range(4,10))]
        v=veg.loc[(veg.index.year==year)&veg.index.month.isin(range(4,10))]
        length, start, end=longest_run(d.precip_daily.to_numpy()<1)
        x=d.index.dayofyear
        axes[0,j].bar(x,d.precip_daily,width=1,color=COLORS[1]);axes[0,j].set_title(str(year))
        axes[1,j].plot(v.index.dayofyear,v.ndvi,marker='o',ms=3,color=COLORS[0])
        for ax in axes[:,j]:
            ax.axvspan(x[start], x[end], color='gray',alpha=.18)
        axes[1,j].set_xlabel('Day of year')
    axes[0,0].set_ylabel('Daily rain (mm)');axes[1,0].set_ylabel('NDVI')
    fig.tight_layout();savefig(root,'04_pair_timeline')

def spell_responses(root, daily, veg):
    """Describe one longest seasonal low-rain spell per year, with censored recovery."""
    records=[]
    for year, group in daily.groupby(daily.index.year):
        d=group.loc[group.index.month.isin(range(4,10))]
        length,i,j=longest_run(d.precip_daily.to_numpy()<1.)
        start,end=d.index[i],d.index[j]
        # Use only composites ending before the spell as the pre-spell reference.
        before=veg.loc[(veg.index+pd.Timedelta(days=7)<start)&(veg.index>=start-pd.Timedelta(days=24))]
        after=veg.loc[(veg.index>end)&(veg.index<=end+pd.Timedelta(days=64))]
        if before.empty or after.empty:
            continue
        pre=before.ndvi.iloc[-1]
        first=after.ndvi.iloc[0]
        change=first-pre
        recovered=after.loc[after.ndvi>=pre]
        recovery=(recovered.index[0]-end).days if change<0 and not recovered.empty else np.nan
        records.append(dict(year=year,start=start.date(),end=end.date(),days=length,
                            pre_ndvi=pre,post_ndvi=first,ndvi_change=change,
                            recovery_days=recovery,decline=change<0,
                            recovery_observed=bool(change<0 and not recovered.empty)))
    events=table(root,'spell_responses',pd.DataFrame(records))
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.5))
    axes[0].scatter(events.days,events.ndvi_change)
    axes[0].axhline(0,color='gray',lw=.8)
    axes[0].set(xlabel='Longest seasonal low-rain run (days)',ylabel='First post-spell NDVI minus pre-spell NDVI',title='One spell per year')
    axes[1].bar(events.year.astype(str),events.ndvi_change,color=[COLORS[1] if x<0 else COLORS[0] for x in events.ndvi_change])
    axes[1].axhline(0,color='gray',lw=.8)
    axes[1].set(ylabel='NDVI change',xlabel='Year',title='Direction of the observed change')
    axes[1].tick_params(axis='x',rotation=90,labelsize=7)
    fig.tight_layout();savefig(root,'05_spell_responses')
    return events

def feature_frame(daily, veg, window=32, threshold=1.):
    records=[]
    for t, row in veg.iterrows():
        if t.month not in range(4,10):
            continue
        # Strictly antecedent daily values; no centered windows or future rain.
        d=daily.loc[(daily.index>=t-pd.Timedelta(days=window))&(daily.index<t)]
        if len(d)!=window:
            continue
        prior=veg.loc[veg.index+pd.Timedelta(days=7)<t]
        if prior.empty or (t-prior.index[-1]).days>24:
            continue
        angle=2*np.pi*(t.dayofyear-1)/365.25
        records.append(dict(date=t,year=t.year,ndvi=row.ndvi,
             sin_doy=np.sin(angle),cos_doy=np.cos(angle),sin2=np.sin(2*angle),cos2=np.cos(2*angle),
             trend=t.year-2001,temp=d.temp_c.mean(),previous_ndvi=prior.ndvi.iloc[-1],
             feature_last_date=d.index[-1],**rhythm(d.precip_daily,threshold)))
    return pd.DataFrame(records)

SEASON=['sin_doy','cos_doy','sin2','cos2','trend','temp']
MODELS={'Season + temperature':SEASON,
        'Rain amount':SEASON+['rain_total'],
        'Rain timing':SEASON+['rain_total','wet_days','dry_run','concentration'],
        'Amount + prior NDVI':SEASON+['rain_total','previous_ndvi'],
        'Timing + prior NDVI':SEASON+['rain_total','wet_days','dry_run','concentration','previous_ndvi'],
        'Persistence':[]}

def evaluate(frame, alpha=10., first_test_year=2010):
    """Expanding-year evaluation: each test year uses earlier years only."""
    predictions=[]
    for year in sorted(frame.year.unique()):
        if year<first_test_year:
            continue
        train=frame.loc[frame.year<year];test=frame.loc[frame.year==year]
        if len(train)<50 or test.empty:
            continue
        assert train.year.max()<test.year.min()
        for name,columns in MODELS.items():
            if name=='Persistence':
                pred=test.previous_ndvi.to_numpy()
            else:
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
                model.fit(train[columns],train.ndvi)
                pred=model.predict(test[columns])
            predictions.extend(dict(date=t,year=year,model=name,observed=y,predicted=p,
                                    training_last_year=int(train.year.max()))
                               for t,y,p in zip(test.date,test.ndvi,pred))
    pred=pd.DataFrame(predictions)
    fold=[]
    for (name,year),g in pred.groupby(['model','year']):
        fold.append(dict(model=name,year=year,n=len(g),mae=mean_absolute_error(g.observed,g.predicted),
                         rmse=np.sqrt(mean_squared_error(g.observed,g.predicted))))
    scores=[]
    for name,g in pred.groupby('model'):
        scores.append(dict(model=name,n=len(g),mae=mean_absolute_error(g.observed,g.predicted),
                           rmse=np.sqrt(mean_squared_error(g.observed,g.predicted)),r2=r2_score(g.observed,g.predicted)))
    return pred,pd.DataFrame(fold),pd.DataFrame(scores)

def plot_evaluation(root,pred,fold,scores):
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.6))
    order=list(MODELS)
    s=scores.set_index('model').loc[order]
    axes[0].bar(range(len(order)),s.mae,color=COLORS+[COLORS[0],COLORS[2]])
    axes[0].set_xticks(range(len(order)),['Season +\ntemperature','Rain\namount','Rain\ntiming','Amount +\nprior NDVI','Timing +\nprior NDVI','Persistence'])
    axes[0].tick_params(axis='x', labelsize=7)
    axes[0].set(ylabel='Mean absolute error (NDVI)',title='All held-out observations')
    for name in order:
        f=fold.loc[fold.model==name]
        axes[1].plot(f.year,f.mae,marker='o',ms=3,label=name)
    axes[1].set(xlabel='Held-out year',ylabel='Mean absolute error',title='Year-to-year performance')
    axes[1].legend(fontsize=7)
    fig.tight_layout();savefig(root,'06_prediction')

def sensitivity(root,daily,veg,annual):
    settings=[]
    for window,threshold in [(16,1.),(32,.5),(32,1.),(32,2.),(64,1.)]:
        f=feature_frame(daily,veg,window,threshold)
        _,_,s=evaluate(f)
        s=s.set_index('model')
        settings.append(dict(window=window,threshold=threshold,n=int(s.loc['Rain timing','n']),
                  amount_mae=s.loc['Rain amount','mae'],timing_mae=s.loc['Rain timing','mae'],
                  improvement=s.loc['Rain amount','mae']-s.loc['Rain timing','mae']))
    result=table(root,'sensitivity',pd.DataFrame(settings))
    matches=table(root,'matching_sensitivity',pd.DataFrame([dict(tolerance=t,pairs=len(match_years(annual,t))) for t in [.05,.10,.15]]))
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.4))
    labels=[f"{int(x.window)} d / {x.threshold:g} mm" for x in result.itertuples()]
    axes[0].barh(labels,result.improvement,color=[COLORS[0] if x>0 else COLORS[1] for x in result.improvement])
    axes[0].axvline(0,color='gray',lw=.8)
    axes[0].set(xlabel='Amount MAE minus timing MAE',title='Antecedent window / wet-day threshold')
    axes[1].bar([f'{int(x*100)}%' for x in matches.tolerance],matches.pairs,color=COLORS[2])
    axes[1].set(xlabel='Annual rainfall matching tolerance',ylabel='Eligible year pairs',title='Descriptive comparisons')
    fig.tight_layout();savefig(root,'07_sensitivity')
    return result,matches

def summary(root,annual,pairs,pair,events,frame,scores,fold,sens):
    score=scores.set_index('model')
    pivot=fold.pivot(index='year',columns='model',values='mae')
    improvements=pivot['Rain amount']-pivot['Rain timing']
    rng=np.random.default_rng(2026)
    boot=np.array([rng.choice(improvements.to_numpy(),len(improvements),replace=True).mean() for _ in range(5000)])
    out=dict(years=len(annual),year_start=int(annual.year.min()),year_end=int(annual.year.max()),
         annual_rain_min=float(annual.annual_rain.min()),annual_rain_max=float(annual.annual_rain.max()),
         ndvi_mean_min=float(annual.ndvi_mean.min()),ndvi_mean_max=float(annual.ndvi_mean.max()),
         seasonal_observations=len(frame),pairs=len(pairs),pair=pair.to_dict(),
         negative_pairs=int((pairs.ndvi_difference<0).sum()),
         event_count=len(events),declines=int(events.decline.sum()),recoveries=int(events.recovery_observed.sum()),
         median_spell=float(events.days.median()),max_spell=int(events.days.max()),
         model_scores=scores.to_dict('records'),test_years=len(pivot),
         improved_years=int((improvements>0).sum()),mean_year_mae_improvement=float(improvements.mean()),
         improvement_interval=np.quantile(boot,[.025,.975]).tolist(),
         pooled_relative_improvement=float(1-score.loc['Rain timing','mae']/score.loc['Rain amount','mae']),
         sensitivity_positive=int((sens.improvement>0).sum()))
    (Path(root)/'outputs/summary.json').write_text(json.dumps(out,indent=2))
    return out
