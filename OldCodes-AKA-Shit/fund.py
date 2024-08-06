# 还是pyspark版，年代太过久远不记得这页代码的意义所在了

# %%
w.isconnected()

# %%
# DT = spark.sql('select date_sub(current_date, 1)').first()[0]                           # T-1天
DT = date(2022,1,11)
BDT = DT - timedelta(days = 364)
FT = date(2021,1,1)

# %%
# 一年国债到期收益率
riskfree = spark.createDataFrame(
    w.edb("S0059744", FT, DT, usedf=True)[1].reset_index()
    ,['date', 'riskfree'])
# 灵活配置型基金指数收益率
benchmark = spark.createDataFrame(
    w.wsd("885061.WI", "pct_chg", FT, DT, "", usedf=True)[1].reset_index()
    ,['date', 'benchmark'])
# 基金净值&收益率
nav000126 = spark.createDataFrame(
    w.wsd("000126.OF", "nav,NAV_acc,NAV_adj,NAV_adj_return1", FT, DT, "", usedf=True)[1].reset_index()
    ,['date', 'nav', 'acc_nav', 'adj_nav', 'adj_pctchg'])

# %%
nav = nav000126.dropna(subset = ['nav']).fillna(0.0, subset = ['adj_pctchg'])
nav = nav.join(benchmark, 'date').join(riskfree, 'date').withColumn('code', F.expr('"000126.OF"'))
nav = nav.withColumn('riskfree', ((1 + F.col('riskfree') / 100) ** ( 1 / 250.0) - 1) * 100)

# %%
def daily_return(nav):
    window = Window.partitionBy('code').orderBy('date')
    return nav.withColumn(
        'return',
        F.round((F.col('adj_nav') / F.lag('adj_nav').over(window) - 1) * 100, 100)
        ).sort(nav.date.desc())
daily_return(nav).show(5)

# %%
((30.10 / 100 + 1) ** (250 / 244.0) - 1) * 100

# %%
def period_return(nav, bgndt=BDT, enddt=DT): #截止日期默认为最新日期
    nav = nav.filter(nav.date.between(bgndt, enddt))    #截取给定时间区间范围内的数据
    mul_udf = F.udf(lambda x: reduce(mul, x))   # udf, 用以实现累乘
    col = F.col('adj_pctchg') / 100 + 1             # 收益率百分号与1的处理
    benchcol = F.col('benchmark') / 100 + 1
    riskcol = F.col('riskfree') / 100 + 1
    nav = nav.groupBy('code').agg(
        ((mul_udf(F.collect_list(col)) - 1) * 100).alias('returnmn'),
        ((mul_udf(F.collect_list(benchcol)) - 1) * 100).alias('bench_returnmn'),
        ((mul_udf(F.collect_list(riskcol)) - 1) * 100).alias('risk_returnmn'),
        F.count('date').alias('days')).select(
            'code',
            'returnmn', 
            'bench_returnmn',
            (F.col('returnmn') - F.col('bench_returnmn')).alias('active_returnmn'),
            ((F.col('returnmn') / 100 + 1) ** (250.0 / (F.col('days') + 1)) * 100 - 100).alias('ann_returnmn'), # 奇怪，为什么要+1，但是不+1和平台对不上
            ((F.col('bench_returnmn') / 100 + 1) ** (250.0 / F.col('days')) * 100 - 100).alias('ann_benchmark'),
            ((F.col('risk_returnmn') / 100 + 1) ** (250.0 / F.col('days')) * 100 - 100).alias('ann_riskfree')
            )
    return nav.select(
        'code',
        F.round('returnmn', 2).alias('returnmn'),
        F.round('ann_returnmn', 2).alias('ann_returnmn'),
        F.round('active_returnmn', 2).alias('active_returnmn'),
        F.round(F.col('ann_returnmn') - F.col('ann_benchmark'), 2).alias('ann_active'),
        F.round('bench_returnmn', 2).alias('bench_returnmn'),
        F.round('ann_benchmark', 2).alias('ann_benchmark'),
        F.round('ann_riskfree', 2).alias('ann_riskfree')
        )
period_return(nav).show()

# %%
def beta_alpha(nav, bgndt=BDT, enddt=DT):
    nav = nav.filter(nav.date.between(bgndt, enddt))
    mul_udf = F.udf(lambda x: reduce(mul, x))
    rdds = nav.groupBy('code').agg(
        F.collect_list('adj_pctchg').alias('returnmn'),
        F.collect_list('riskfree').alias('riskfreemn'),
        F.collect_list('benchmark').alias('benchmarkmn')
        ).rdd.map(lambda nav : ({nav.code: [nav.returnmn, nav.riskfreemn, nav.benchmarkmn]}))
    def beta(rdd):
        codes = list(rdd.keys())[0]
        values = rdd.get(codes)
        ret = np.array(values[0])
        risk = np.array(values[1])
        bench = np.array(values[2])
        cov = np.cov(ret - risk, bench - risk)
        beta = cov[0][1] / cov[1][1]
        return codes, float(beta)
    df = rdds.map(beta).toDF(['code', 'beta'])
    nav = nav.join(df, 'code')
    col = F.expr('adj_pctchg - riskfree - beta * (benchmark - riskfree)')
    nav = nav.groupBy('code').agg(
        F.max('beta').alias('beta'),
        F.mean(col).alias('alpha')
        ).withColumn('alpha', F.expr('250 * alpha / 100'))
    return nav.select(
        'code',
        F.round('beta', 2).alias('beta'),
        F.round('alpha', 2).alias('alpha')
        )
beta_alpha(nav).show()

# %%
def new_high(nav, bgndt=BDT, enddt=DT):
    top = nav.filter(
        nav.date <= bgndt       # 区间外，选定时间段之前的时间
        ).groupBy('code').agg(
            F.max('nav').alias('top_nav'))  # 历史最大净值，作为外沿新高率的判断基准
    navs = nav.filter(
        nav.date.between(bgndt, enddt)
        ).sort(nav.date).groupBy('code').agg(
            F.collect_list('nav').alias('navs')
            )   # 选定时间内，按基金分组，按时间排序处理成净值数组
    rdds = navs.join(top, 'code', 'left').fillna(0.0, subset=['top_nav']
        ).rdd.map(lambda nav: ({nav.code: [nav.navs, nav.top_nav]}))    # RDD化
    def _new_high(rdd):
        # 计算单支基金净值新高率的函数
        codes = list(rdd.keys())[0]     # 取基金代码
        values = rdd.get(codes)
        days = len(values[0])          # 总天数
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
        days = float(days) + 1      # 2022-01-12：奇怪，这里也多+1了，财富文档修改以前，这里是不用+1的，他们改了逻辑，不明白为什么
        return codes, round(inner_break / days * 100, 2), round(outer_break / days * 100, 2)
        # 内含新高率，以第一天作为基准，因此计算新高率时，第一天应排除在外，所以是days-1
    return rdds.map(_new_high).toDF(['code', 'inner_nhr', 'outer_nhr']) # 在rdd集上应用函数并重新转化为DataFrame
new_high(nav).show()

# %%
def profit_decline(nav, bgndt=BDT, enddt=DT):
    return nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.count('date').alias('total'),
        F.count(F.when(nav.adj_pctchg > 0, 'date')).alias('profit'),
        F.count(F.when(nav.adj_pctchg < 0, 'date')).alias('decline')
        ).withColumn('dec_pct', F.expr('round(decline / (total+1) * 100, 2)')
        ).withColumn('pft_pct', F.expr('round(profit / (total+1) * 100, 2)'))   # 这里也+1了，就离谱，加起来都没有100了呀！
profit_decline(nav).show()

# %%
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
            'roll',
            F.round('roll', 2).alias('roll_ret')
            )
    latest_rollret = nav.select('roll').first()[0]
    roll_gp = nav.groupBy('code').agg(
        F.first('roll_ret').alias('latest_rollret'),
        F.round(F.count(F.when(F.col('roll') < latest_rollret, 1)) / F.count('roll') * 100, 2).alias('history'),
        F.round(F.count(F.when(F.col('roll') > 0, 'date')) / F.count('date') * 100, 2).alias('profit_rate'),
        F.max('roll_ret').alias('max_rollret'),
        F.min('roll_ret').alias('min_rollret')
        )
    return nav.drop('roll'), roll_gp
roll_halfy = roll(nav, 6)
roll_halfy[1].show()
roll_halfy[0].show(5)

# %%
def sharpe(nav, bgndt=BDT, enddt=DT):
    nav = nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.stddev('adj_pctchg').alias('vol'),
        F.mean(F.expr('adj_pctchg - riskfree')).alias('avg_riskadj')
        ).withColumn(
            'ann_vol',
            F.expr('vol * sqrt(249)')   # 这里又变成了-1，真是怪
            ).withColumn(
                'sharpe',
                sqrt(250) * F.col('avg_riskadj') / F.col('vol')
                )
    return nav.select(
        'code',
        F.round('vol', 2).alias('vol'),
        F.round('ann_vol', 2).alias('ann_vol'),
        F.round('sharpe', 2).alias('sharpe')
        )
vol(nav).show()

# %%
def sortino(nav, bgndt=BDT, enddt=DT):
    nav.filter(nav.date.between(bgndt, enddt))
    negative = nav.filter('adj_pctchg < 0').groupBy('code').agg(
        F.stddev_pop('adj_pctchg').alias('std'),
        F.count('date').alias('cnt')
        ).select(
            'code',
            F.expr('std * sqrt(cnt)').alias('negative')
            )
    nav = nav.groupBy('code').agg(
        F.mean(F.expr('adj_pctchg - riskfree')).alias('avg_riskadj'),
        F.count('date').alias('cnt')
        ).join(negative, 'code').withColumn(
            'down_risk',
            F.expr('negative / sqrt(cnt - 1)') * sqrt(250)
            ).withColumn(
                'sortino',
                F.expr('avg_riskadj / down_risk') * 250
                )
    return nav.select(
            'code',
            F.round('down_risk', 2).alias('down_risk'),
            F.round('sortino', 2).alias('sortino')
        )
sortino_sharpe(nav).show()  # 下行波动差距很小，但索提诺仍有点gap，多半是无风险收益率不正确引起的

# %%
def track(nav, bgndt=BDT, enddt=DT):
    active = (F.col('adj_pctchg') - F.col('benchmark')) / 100
    nav = nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.stddev(active).alias('std'),
        F.mean(active).alias('mean')
        )
    return nav.withColumn(
        'track_error',
        F.expr('std * sqrt(250) * 100')
        ).withColumn(
            'info',
            F.expr('mean / track_error * 250 * 100')
            ).select(
                'code',
                F.round('track_error', 2).alias('track_error'),
                F.round('info', 2).alias('info')
                )
track(nav).show()

# %%
def drawdown(nav, bgndt=BDT, enddt=DT):
    # 净值&主动收益回撤曲线
    nav = nav.filter(nav.date.between(bgndt, enddt))
    window = Window.partitionBy('code').orderBy('date')
    mul_udf = F.udf(lambda x: reduce(mul, x))   # udf, 用以实现累乘
    col = F.col('adj_pctchg') / 100 + 1             # 收益率百分号与1的处理
    col2 = (F.col('adj_pctchg') - F.col('benchmark')) / 100 + 1  # 主动收益率
    nav = nav.dropna(subset = ['adj_pctchg']).withColumn(
        'returnmnt',
        mul_udf(F.collect_list(col).over(window))
        ).withColumn(
            'ac_returnmnt',
            mul_udf(F.collect_list(col2).over(window))
            )
    nav = nav.withColumn(
        'drawdown',
        F.round((1 - F.col('returnmnt') / F.max('returnmnt').over(window)) * 100, 4)
        ).withColumn(
            'ac_drawdown',
            F.round((1 - F.col('ac_returnmnt') / F.max('ac_returnmnt').over(window)) * 100, 4)
            ).sort(nav.date.desc())
    return nav.select(
        'code',
        'date',
        F.round('adj_nav', 6).alias('adj_nav'),
        F.round('returnmnt', 2).alias('returnmnt'),
        F.round('ac_returnmnt', 2).alias('ac_returnmnt'),
        F.round('drawdown', 2).alias('drawdown'),
        F.round('ac_drawdown', 2).alias('ac_drawdown')
    )
nav_dd = drawdown(nav)
nav_dd.show(5)

# %%
def max_drawdown(nav_dd):
    return nav_dd.groupBy('code').agg(
        F.max('drawdown').alias('drawdown'),
        F.max('ac_drawdown').alias('ac_drawdown')
        )
max_drawdown(nav_dd).show()

# %%
def drawdown_recover(nav_dd, bgndt=BDT, enddt=DT):
    max_dt = nav_dd.join(
        max_drawdown(nav_dd),
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
            ).filter(nav_dd.date.between(max_dt.max_dt, rec_dt.rec_dt)).groupBy('code').agg(
                (F.count('date') - 1).alias('recovery_time')
            )
drawdown_recover(drawdown(nav)).show()

# %%
def max_decline(nav, bgndt=BDT, enddt=DT):
    return nav.filter(nav.date.between(bgndt, enddt)).groupBy('code').agg(
        F.round(F.min('adj_pctchg'), 2).alias('max_decline')
        )
max_decline(nav).show()

# %%
def keep_decline(nav, bgndt=BDT, enddt=DT):
    nav = nav.filter(nav.date.between(bgndt, enddt))
    window = Window.partitionBy('code').orderBy('date')
    nav = nav.select(
        'code',
        'date',
        'adj_pctchg',
        F.row_number().over(window).alias('rn'),
        F.lag('adj_pctchg').over(window).alias('ret_lag')
        ).withColumn(
            'decline',
            F.sum(F.expr('if(adj_pctchg < 0 and ret_lag < 0, 1, 0)')).over(window)
            ).withColumn('diff', F.expr('rn - decline'))
    return nav.groupBy('code', 'diff').agg(
        F.count('diff').alias('keep_decline')
        ).groupBy('code').agg(F.max('keep_decline').alias('max_keep_decline'))
keep_decline(nav).show()

# %%
def var(nav, bgndt=BDT, enddt=DT):
    nav = nav.filter(nav.date.between(bgndt, enddt))
    return nav.groupBy('code').agg(
        F.round(F.percentile_approx('adj_pctchg', 0.05), 2).alias('VaR')
        )
var(nav).show()

# %%
def calmar(nav, bgndt=BDT, enddt=DT):
    ann = period_return(nav, bgndt, enddt)
    dd = max_drawdown(drawdown(nav, bgndt, enddt))
    return ann.join(dd, 'code').select(
        'code',
        F.round(F.expr('(ann_returnmn - ann_riskfree) / drawdown'), 2).alias('calmar')      # !遗漏之处：文档中，卡玛比率的计算方式也发生了变更，已按照新公式验证无误
        )
calmar(nav).show()

# %%
def profit_quotas(nav, bgndt=BDT, enddt=DT):
    df1 = period_return(nav, bgndt, enddt).select('code', 'returnmn', 'ann_returnmn', 'active_returnmn')
    df2 = new_high(nav, bgndt, enddt)
    df3 = profit_decline(nav, bgndt, enddt).select('code', 'profit', 'pft_pct')
    df4 = beta_alpha(nav, bgndt, enddt).select('code', 'alpha', 'beta')
    df1.join(df2, 'code').join(df3, 'code').join(df4, 'code').show()
profit_quotas(nav, DT - timedelta(days = 179))

# %%
def risk_quotas(nav, bgndt=BDT, enddt=DT):
    df1 = sharpe(nav, bgndt, enddt).select('code', 'vol', 'ann_vol')
    df2 = sortino(nav, bgndt, enddt).select('code', 'down_risk')
    df3 = max_drawdown(drawdown(nav, bgndt, enddt))
    drawdown_recover(drawdown(nav, bgndt, enddt)).show()
    df5 = max_decline(nav, bgndt, enddt)
    df6 = keep_decline(nav, bgndt, enddt)
    df7 = profit_decline(nav, bgndt, enddt).select('code', 'dec_pct')
    df8 = track(nav, bgndt, enddt).select('code', 'track_error')
    df9 = var(nav, bgndt, enddt)
    df1.join(df2, 'code').join(df3, 'code').join(df5, 'code').join(df6, 'code'
        ).join(df7, 'code').join(df8, 'code').join(df9, 'code').show()
risk_quotas(nav, DT -timedelta(days = 364)) # 下行风险仍然有较小的gap

# %%
def risk_adjusted_quotas(nav, bgndt=BDT, enddt=DT):
    df1 = sharpe(nav, bgndt, enddt).select('code', 'sharpe')
    df2 = track(nav, bgndt, enddt).select('code', 'info')
    df3 = sortino(nav, bgndt, enddt).select('code', 'sortino')
    df4 = calmar(nav, bgndt, enddt)
    df1.join(df2, 'code').join(df3, 'code').join(df4, 'code').show()
risk_adjusted_quotas(nav, DT - timedelta(days = 364))   # 索提诺误差最大，尤其周期较短时。夏普也有一点


