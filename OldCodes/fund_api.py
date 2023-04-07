def daily_return(nav):
    window = Window.partitionBy('code').orderBy('date')
    return nav.withColumn(
        'return',
        F.round((F.col('nav') / F.lag('nav').over(window) - 1) * 100, 4)
        ).sort(nav.date.desc())
daily_return(nav).show(5)


def period_return(nav, bgndt=BDT, enddt=DT): #截止日期默认为最新日期
    nav = nav.filter(nav.date.between(bgndt, enddt))    #截取给定时间区间范围内的数据
    mul_udf = F.udf(lambda x: reduce(mul, x))   # udf, 用以实现累乘
    col = F.col('nav_pctchg') / 100 + 1             # 收益率百分号与1的处理
    benchcol = F.col('benchmark') / 100 + 1
    return nav.groupBy('code').agg(
        F.round((mul_udf(F.collect_list(col)) - 1) * 100, 2).alias('returnmn'),
        F.round((mul_udf(F.collect_list(benchcol)) - 1) * 100, 2).alias('bench_returnmn'),
        F.max('days_per_year').alias('days_per_year'),
        F.count('date').alias('days')).withColumn(
            'ann_returnmn',
            F.round((F.col('returnmn') / 100 + 1) ** (F.col('days_per_year') / F.col('days')) * 100 - 100, 2)
            ).withColumn(
                'active_returnmn',
                F.col('returnmn') - F.col('bench_returnmn')
                ).drop('days', 'days_per_year', 'bench_returnmn')
period_return(nav, '2020-01-01', '2020-12-31').show()


def new_high(nav, bgndt=BDT2, enddt=DT):
    top = nav.filter(
        nav.date <= bgndt       # 区间外，选定时间段之前的时间
        ).groupBy('code').agg(
            F.max('nav').alias('top_nav'))  # 历史最大净值，作为外沿新高率的判断基准
    navs = nav.filter(
        nav.date.between(bgndt, enddt)
        ).sort(nav.date).groupBy('code').agg(
            F.collect_list('nav').alias('navs')
            )   # 选定时间内，按基金分组，按时间排序处理成净值数组
    rdds = top.join(navs, 'code').rdd.map(lambda nav: ({nav.code: [nav.navs, nav.top_nav]}))    # RDD化
    def _new_high(rdd):
        # 计算单支基金净值新高率的函数
        codes = list(rdd.keys())[0]     # 取基金代码
        values = rdd.get(codes)
        days = len(values[0])           # 总天数
        bgn_nav = values[0][0]          # 区间内第一天的净值，作为内含新高率的判断基准
        top_nav = values[1]             # 外沿历史最高净值，在RDD化的操作中join进来作为了
        inner_break = 0         # 记录创新高的次数
        outer_break = 0
        for i in range(0, days):
            temp = values[0][i]     # 存储第i天的净值
            if temp > top_nav:
                outer_break += 1
                top_nav = temp      # 外沿创新高，用新高替代历史最高，创新高次数+1
            if temp > bgn_nav:
                inner_break += 1
                bgn_nav = temp      # 内含创新高，用新高替代初始净值，创新高次数+1
        return codes, round(inner_break / float(days - 1) * 100, 2), round(outer_break / float(days) * 100, 2)
        # 内含新高率，以第一天作为基准，因此计算新高率时，第一天应排除在外，所以是days-1
    return rdds.map(_new_high).toDF(['code', 'inner_nhr', 'outer_nhr']) # 在rdd集上应用函数并重新转化为DataFrame
new_high(nav).show()


def profit_decline(nav, bgndt=BDT, enddt=DT):
    return nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.count('date').alias('total'),
        F.count(F.when(nav.nav_pctchg > 0, 'date')).alias('profit'),
        F.count(F.when(nav.nav_pctchg < 0, 'date')).alias('decline')
        ).withColumn('dec_pct', F.expr('round(decline / total * 100, 2)'))
profit_decline(nav, '2020-11-19').show()


def roll(nav, time, date=DT):
    a = nav.withColumn(
        'date', 
        F.expr('date_add(add_months(date, -{}), 0)'.format(time))
        ).select(
            'code',
            'date', 
            F.col('adj_nav').alias('roll_nav')
            )
    nav = nav.join(
        a,
        ['date', 'code'],
        'left').withColumn(
            'roll_nav',
            F.last('roll_nav', ignorenulls=True).over(Window.partitionBy('code').orderBy('date'))
            )
    nav = nav.sort(nav.date.desc()).filter('date <= date_add(add_months("{0}", -{1}), 0)'.format(date, time)).withColumn(
        'roll', 
        F.expr('roll_nav / adj_nav - 1') * 100
        ).select(
            'code',
            'date',
            F.round('roll', 2).alias('roll_ret')
            )
    latest_rollret = nav.select('roll_ret').first()[0]
    roll_gp = nav.groupBy('code').agg(
        F.first('roll_ret').alias('latest_rollret'),
        F.round(F.count(F.when(F.col('roll_ret') < latest_rollret, 1)) / F.count('roll_ret') * 100, 2).alias('history'),
        F.round(F.count(F.when(F.col('roll_ret') > 0, 'date')) / F.count('date') * 100, 2).alias('profit_rate'),
        F.max('roll_ret').alias('max_rollret'),
        F.min('roll_ret').alias('min_rollret')
        )
    return nav, roll_gp
roll_halfy = roll(nav, 6)
roll_halfy[1].show()
roll_halfy[0].show()


def vol(nav, bgndt=BDT, enddt=DT):
    return nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.round(F.stddev_pop('nav_pctchg'), 4).alias('vol'),
        F.max('days_per_year').alias('days_per_year')
        ).withColumn(
            'ann_vol',
            F.expr('round(vol * sqrt(days_per_year), 4)')
            ).drop('days_per_year')
vol(nav, '2020-01-01', '2020-12-31').show()


30.6902


def sortino_sharpe(nav, bgndt=BDT2, enddt=DT):
    adj_risk = F.col('nav_pctchg') - F.col('riskfree')
    adj_risk0 = F.when(adj_risk > 0 , 0).otherwise(adj_risk)
    nav = nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.sum(adj_risk0 ** 2).alias('sum'),
        F.mean(adj_risk).alias('mean'),
        F.mean('nav_pctchg').alias('mean2'),
        F.stddev(adj_risk).alias('std'),
        F.count('date').alias('cnt'),
        F.max('days_per_year').alias('dpy')
        )
    return nav.select('mean2')
    return nav.withColumn(
            'down_vol',
            F.expr('sqrt(dpy * sum / (cnt - 1))')
            ).withColumn(
                'sortino',
                F.expr('dpy * mean / down_vol')
                ).withColumn(
                    'sharpe',
                    F.expr('sqrt(dpy) * mean / std')
                ).select(
                    'code',
                    F.round('down_vol', 2).alias('down_vol'),
                    F.round('sortino', 4).alias('sortino'),
                    F.round('sharpe', 4).alias('sharpe')
                    )
sortino_sharpe(nav).show()


def track(nav, bgndt=BDT, enddt=DT):
    active = (F.col('nav_pctchg') - F.col('benchmark')) / 100
    nav = nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.stddev_pop(active).alias('std'),
        F.mean(active).alias('mean'),
        F.max('days_per_year').alias('dpy')
        )
    return nav.withColumn(
        'track_error',
        F.expr('std * sqrt(dpy)')
        ).withColumn(
            'info',
            F.expr('mean / track_error * dpy')
            ).select(
                'code',
                F.round('track_error', 4).alias('track_error'),
                F.round('info', 4).alias('info')
                )
track(nav, '2019-12-31', '2020-12-31').show()


def drawdown(nav):
    # 回撤曲线
    window = Window.partitionBy('code').orderBy('date')
    return nav.withColumn(
        'drawdown',
        F.round((1 - F.col('adj_nav') / F.max('adj_nav').over(window)) * 100, 4)
        ).sort(nav.date.desc())
nav_dd = drawdown(nav)
nav_dd.show()


def max_drawdown(nav_dd, bgndt=BDT, enddt=DT):
    return nav_dd.filter(nav_dd.date.between(bgndt, enddt)).groupBy('code').agg(
        F.max('drawdown').alias('drawdown')
        )
max_drawdown(nav_dd, '2020-01-01', '2020-12-31').show()


def drawdown_recover(nav_dd, bgndt=BDT, enddt=DT):
    max_dt = nav_dd.join(
        max_drawdown(nav_dd, bgndt, enddt),
        ['code', 'drawdown']).select(
            'code',
            F.col('adj_nav').alias('max_adj_nav'), 
            F.col('date').alias('max_dt'),
            F.col('drawdown').alias('max_drawdown')
            )
    rec_dt = nav_dd.join(
        max_dt,
        'code'
        ).filter(nav_dd.date > max_dt.max_dt).groupBy('code').agg(
           F.min(F.when(
                F.expr('drawdown = 0'), F.col('date'))).alias('rec_dt')
            )
    return nav_dd.join(
        max_dt,
        'code'
        ).join(
            rec_dt,
            'code'
            ).filter(nav.date.between(max_dt.max_dt, rec_dt.rec_dt)).groupBy('code').count()
drawdown_recover(nav_dd).show()


def ac_drawdown(nav):
    # 主动收益回撤曲线
    window = Window.partitionBy('code').orderBy('date')
    nav = nav.withColumn(
        'active_nav',
        F.expr('nav / (100 + nav_pctchg) * (100 + nav_pctchg - benchmark)')
        )
    return nav.withColumn(
        'drawdown',
        F.round((1 - F.col('active_nav') / F.max('active_nav').over(window)) * 100, 4)
        ).sort(nav.date.desc())
max_drawdown(ac_drawdown(nav)).show()


def max_decline(nav, bgndt=BDT, enddt=DT):
    return nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.min('nav_pctchg').alias('max_decline')
        )
max_decline(nav).show()


def keep_decline(nav, bgndt=BDT, enddt=DT):
    window = Window.partitionBy('code').orderBy('date')
    nav = nav.select(
        'code',
        'date',
        'nav_pctchg',
        F.row_number().over(window).alias('rn'),
        F.lag('nav_pctchg').over(window).alias('ret_lag')
        ).withColumn(
            'decline',
            F.sum(F.expr('if(nav_pctchg < 0 and ret_lag < 0, 1, 0)')).over(window)
            ).withColumn('diff', F.expr('rn - decline'))
    return nav.groupBy('code', 'diff').agg(
        F.count('diff').alias('keep_decline')
        ).groupBy('code').agg(F.max('keep_decline').alias('max_keep_decline'))
keep_decline(nav).show()


def var(nav, bgndt=BDT, enddt=DT):
    nav = nav.filter(nav.date.between(bgndt, enddt))
    return nav.groupBy('code').agg(
        (F.percentile_approx('nav_pctchg', 0.05) / 100).alias('VaR')
        )
var(nav, '2018-12-31', '2019-12-31').show()


def calmar(nav, bgndt=BDT, enddt=DT):
    window = Window.partitionBy('code')
    mean = nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.mean('nav_pctchg').alias('mean1'),
        F.mean('riskfree').alias('mean2')
    )
    dd = max_drawdown(drawdown(nav), bgndt, enddt)
    return mean.join(dd, 'code').select(
        'code',
        F.expr('(mean1 - mean2) / drawdown * 100').alias('calmar')
        )
calmar(nav).show()


def beta_alpha(nav, bgndt=BDT, enddt=DT):
    nav = nav.filter(nav.date.between(bgndt, enddt))
    rdds = nav.groupBy('code').agg(
        F.collect_list('nav_pctchg').alias('returnmn'),
        F.collect_list('riskfree').alias('riskfreemn'),
        F.collect_list('benchmark').alias('benchmarkmn')
        ).rdd.map(lambda nav : ({nav.code: [nav.returnmn, nav.riskfreemn, nav.benchmarkmn]}))
    def beta(rdd):
        codes = list(rdd.keys())[0]
        values = rdd.get(codes)
        ret = np.array(values[0]).astype(float)
        risk = np.array(values[1]).astype(float)
        bench = np.array(values[2]).astype(float)
        cov = np.cov(ret - risk, bench - risk)
        beta = cov[0][1] / cov[1][1]
        return codes, np.float(beta)
    df = rdds.map(beta).toDF(['code', 'beta'])
    nav = nav.join(df, 'code')
    col = F.expr('nav_pctchg - riskfree - beta * (benchmark - riskfree)')
    return nav.groupBy('code').agg(
        F.round(F.max('beta'), 4).alias('beta'),
        F.round(F.mean(col), 4).alias('alpha')
    )
beta_alpha(nav, '2020-01-01', '2020-12-31').show()


