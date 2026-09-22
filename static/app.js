const pages = [
  ['overview', '⌂', '风险总览'],
  ['customer', '♙', '客户审查'],
  ['strategy', '◉', '策略模拟'],
  ['analyst', '✦', 'AI Analyst'],
];
const advancedPages = {models: '模型表现与实验', data: '数据质量', features: '特征管理', monitor: '模型健康 / PSI'};
const state = {page: 'overview', customerId: 903112, filters: {grade: 'all', purpose: 'all', income: 'all', risk: 'all'}, strategyThreshold: .35, decision: {}, overviewSeq: 0};
const $ = id => document.getElementById(id);
const h = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const n = (v, d=0) => v == null || Number.isNaN(Number(v)) ? '—' : Number(v).toLocaleString('zh-CN', {maximumFractionDigits:d, minimumFractionDigits:d});
const pct = (v, d=1) => `${n(Number(v)*100,d)}%`;
const api = async (path, options={}) => {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `请求失败 ${response.status}`);
  return body;
};
const card = (title, subtitle, body) => `<section class="card"><h2>${title}</h2><p class="card-sub">${subtitle}</p>${body}</section>`;
const heading = (eyebrow, title, subtitle, actions='') => `<div class="page-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${subtitle}</p></div><div class="heading-actions">${actions}</div></div>`;
const riskTag = risk => `<span class="risk-badge ${risk==='中风险'?'medium':risk==='低风险'?'low':''}">${h(risk)}</span>`;
const tag = s => `<span class="tag ${s==='正常'?'green':s==='关注'?'orange':s==='漂移'?'red':''}">${h(s)}</span>`;
const metric = (label, value) => `<div class="metric-line"><span>${label}</span><strong>${value}</strong></div>`;

function curve(points, xLabel='假阳性率', yLabel='真正率', color='#0b8b81') {
  const w=540, H=225, l=42, r=15, t=12, b=30;
  const x=v=>l+Math.max(0,Math.min(1,v))*(w-l-r), y=v=>H-b-Math.max(0,Math.min(1,v))*(H-t-b);
  const line=points.map((p,i)=>`${i?'L':'M'}${x(p.x).toFixed(1)} ${y(p.y).toFixed(1)}`).join(' ');
  return `<svg class="chart" viewBox="0 0 ${w} ${H}" preserveAspectRatio="none" role="img" aria-label="${h(yLabel)}曲线">
  ${[0,.25,.5,.75,1].map(v=>`<line x1="${x(v)}" y1="${y(0)}" x2="${x(v)}" y2="${y(1)}" stroke="#edf1f4"/><line x1="${x(0)}" y1="${y(v)}" x2="${x(1)}" y2="${y(v)}" stroke="#edf1f4"/><text x="${x(v)}" y="${H-12}" text-anchor="middle" fill="#96a4b2" font-size="10">${v}</text><text x="${l-8}" y="${y(v)+3}" text-anchor="end" fill="#96a4b2" font-size="10">${v}</text>`).join('')}
  <line x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke="#cbd6dc" stroke-dasharray="4 5"/><path d="${line}" fill="none" stroke="${color}" stroke-width="3"/><text x="${w/2}" y="${H-1}" text-anchor="middle" fill="#8a9aa8" font-size="10">${h(xLabel)}</text></svg>`;
}

function beeswarmChart(summary) {
  const names=summary.feature_importance.slice(0,8).map(x=>x.feature);
  const all=names.flatMap(name=>summary.beeswarm[name].map(x=>Math.abs(x.contribution)));
  const scale=Math.max(...all,.01), width=650, center=330;
  const rows=names.map((name,row)=>{
    const points=summary.beeswarm[name], finite=points.map(x=>x.value).filter(Number.isFinite);
    const min=Math.min(...finite), max=Math.max(...finite), spread=max-min||1;
    return `<text x="4" y="${25+row*39}" font-size="11" fill="#50677b">${h(featureLabel(name))}</text><line x1="145" x2="620" y1="${21+row*39}" y2="${21+row*39}" stroke="#edf1f4"/>${points.map((point,i)=>`<circle cx="${center+point.contribution/scale*230}" cy="${21+row*39+((i*17)%15)-7}" r="3" fill="${point.value==null?'#9daeb9':((point.value-min)/spread>.5?'#e8846b':'#5d9bc5')}" opacity=".62"/>`).join('')}`;
  }).join('');
  return `<svg class="beeswarm-chart" viewBox="0 0 ${width} 340" role="img" aria-label="SHAP 特征贡献散点图"><line x1="${center}" x2="${center}" y1="5" y2="315" stroke="#8398a6" stroke-dasharray="4 4"/>${rows}<text x="145" y="334" font-size="10" fill="#8190a1">降低预测风险 ←</text><text x="445" y="334" font-size="10" fill="#8190a1">→ 提高预测风险</text></svg><p class="note-small">蓝色表示数值较低，橙色表示较高，灰色表示缺失；横轴为概率贡献。类别编码的数值高低不等于业务顺序。</p>`;
}

function renderNav() {
  $('nav').innerHTML = pages.map(([id, icon, label]) => `<button class="nav-item ${state.page===id?'active':''}" data-page="${id}"><span class="nav-icon">${icon}</span>${label}</button>`).join('');
  document.querySelectorAll('[data-page]').forEach(button => button.onclick=()=>navigate(button.dataset.page));
}
function closeAdvanced() { $('advancedMenu').hidden=true; $('advancedToggle').setAttribute('aria-expanded','false'); }
async function navigate(page) {
  state.page=page; renderNav(); closeAdvanced();
  $('breadcrumb').textContent = `${page in advancedPages ? '高级 / 模型与数据' : '业务工作台'} / ${advancedPages[page] || pages.find(p=>p[0]===page)?.[2] || page}`;
  $('content').innerHTML='<div class="loading">正在读取数据…</div>';
  try { await ({overview, customer, strategy, analyst, models, data, features, monitor})[page](); }
  catch (error) { $('content').innerHTML=`<div class="error">${h(error.message)}</div>`; }
}
$('advancedToggle').onclick=()=>{ const menu=$('advancedMenu'); menu.hidden=!menu.hidden; $('advancedToggle').setAttribute('aria-expanded',String(!menu.hidden)); };
document.querySelectorAll('[data-advanced]').forEach(button=>button.onclick=()=>navigate(button.dataset.advanced));
$('modelStatus').onclick=()=>navigate('monitor');

async function overview() {
  const options=await api('/api/risk-filters');
  $('content').innerHTML = heading('CREDIT RISK OVERVIEW','信贷风险总览','谁有风险、有哪些值得关注的线索、下一步查看谁。',`<button class="primary" id="openHighRisk">审查高风险客户 →</button>`)+
    `<div class="filter-bar"><label>客户范围 <select id="filterGrade"><option value="all">全部等级</option>${'ABCDEFG'.split('').map(g=>`<option value="${g}">${g} 级</option>`).join('')}</select></label><label>贷款用途 <select id="filterPurpose"><option value="all">全部用途</option>${options.purposes.map(p=>`<option value="${p}">用途编码 ${p}</option>`).join('')}</select></label><label>收入区间 <select id="filterIncome"><option value="all">全部收入</option><option value="low">≤ 6 万</option><option value="middle">6–12 万</option><option value="high">＞ 12 万</option></select></label><label>风险等级 <select id="filterRisk"><option value="all">全部风险</option><option value="low">低风险</option><option value="medium">中风险</option><option value="high">高风险</option></select></label><button class="secondary" id="resetFilters">重置</button></div><div id="overviewBody"></div>`;
  const ids={grade:'filterGrade',purpose:'filterPurpose',income:'filterIncome',risk:'filterRisk'};
  Object.entries(ids).forEach(([key,id])=>{ $(id).value=state.filters[key]; $(id).onchange=()=>{state.filters[key]=$(id).value;loadOverview();}; });
  $('resetFilters').onclick=()=>{state.filters={grade:'all',purpose:'all',income:'all',risk:'all'};Object.entries(ids).forEach(([key,id])=>$(id).value=state.filters[key]);loadOverview();};
  $('openHighRisk').onclick=()=>{state.filters.risk='high';$('filterRisk').value='high';loadOverview();$('overviewBody').scrollIntoView({behavior:'smooth'});};
  await loadOverview();
}
async function loadOverview() {
  const seq=++state.overviewSeq, box=$('overviewBody'); box.innerHTML='<div class="loading">正在计算筛选结果…</div>';
  const query=new URLSearchParams(state.filters);
  const data=await api('/api/risk-overview?'+query);
  if (seq!==state.overviewSeq || !$('overviewBody')) return;
  const colors={'低风险':'','中风险':'orange','高风险':'red'};
  const distribution=Object.entries(data.distribution).map(([name,count])=>`<div class="bar-row"><span>${name}</span><div class="bar-track"><div class="bar-fill ${colors[name]}" style="width:${data.count?count/data.count*100:0}%"></div></div><span class="bar-value">${n(count)}</span></div>`).join('');
  const factors=data.factors.map(f=>`<div class="factor-row"><div><strong>${h(f.label)}</strong><span>高风险组中位数 ${n(f.high_median,1)} · 当前客群 ${n(f.population_median,1)}</span></div><span class="tag">描述性对照</span></div>`).join('') || '<div class="empty">当前筛选无高风险客户</div>';
  box.innerHTML=`<div class="kpi-grid business-kpis">${[['当前评估客户',n(data.count),'测试集 A · 模型评分'],['高风险客户',n(data.high_count),'预测概率 ≥ 0.35'],['中风险客户',n(data.medium_count),'预测概率 0.15–0.35'],['高风险占比',pct(data.high_share),'在当前筛选范围内']].map(([label,value,foot])=>`<div class="card kpi"><div class="label">${label}</div><span class="value">${value}</span><div class="foot">${foot}</div></div>`).join('')}</div>
  <div class="grid-2">${card('风险等级分布','所有图表同步使用上方筛选条件',distribution+'<p class="note-small">分级阈值仅用于演示；当前高风险阈值 0.35。</p>')}${card('值得关注的客群特征','高风险组与相同业务筛选范围的整体客群对照',factors+'<p class="note-small">这些差异不是模型贡献或因果解释。</p>')}</div>
  ${card('优先审查的客户','按模型预测违约概率排序 · 点击客户进入审查',`<div class="table-wrap"><table><thead><tr><th>客户 ID</th><th>预测违约概率</th><th>信用等级</th><th>关注线索</th><th></th></tr></thead><tbody>${data.top_customers.map(row=>`<tr class="clickable" data-customer="${row.id}"><td><strong>#${row.id}</strong></td><td>${pct(row.probability,2)}</td><td>${h(row.grade)}</td><td>${row.clues.map(h).join(' / ')}</td><td>查看 →</td></tr>`).join('')||'<tr><td colspan="5">当前条件无客户</td></tr>'}</tbody></table></div>`)}
  <div class="model-foot"><span id="overviewModelHealth">正在读取模型技术指标…</span><button class="text-button" id="modelDetail">查看模型详情 →</button></div><p class="note-small">${h(data.basis)}</p>`;
  document.querySelectorAll('[data-customer]').forEach(row=>row.onclick=()=>{state.customerId=Number(row.dataset.customer);navigate('customer');});
  $('modelDetail').onclick=()=>navigate('models');
  const [health, model]=await Promise.all([api('/api/monitor'),api('/api/models')]);
  const driftCount=health.drift.filter(x=>x.status==='漂移').length;
  const status=driftCount?`${driftCount} 项特征漂移`:'正常';
  $('modelStatus').textContent=`● 输入分布：${status}`;
  if (seq===state.overviewSeq && $('overviewModelHealth')) $('overviewModelHealth').textContent=`● 输入分布：${status} · 验证 AUC ${n(model.baseline.auc,3)}`;
}

const featureNames={interestRate:'贷款利率',dti:'债务收入比',ficoRangeLow:'FICO 下限',loan_to_income:'贷款收入比',annualIncome:'年收入',loanAmnt:'贷款金额',credit_history_months:'信用历史',revolUtil:'循环额度利用率',term:'贷款期限',grade:'信用等级',subGrade:'细分信用等级',employmentLength:'工作年限',homeOwnership:'住房情况',installment:'月供',issue_year:'贷款发放年份',issue_month:'贷款发放月份'};
const businessFeatures=new Set(Object.keys(featureNames));
const featureLabel = key => featureNames[key] || key;
const pp = value => `${value>=0?'+':'−'}${n(Math.abs(value)*100,2)} 个百分点`;
async function customer() {
  $('content').innerHTML=heading('CUSTOMER REVIEW','客户审查','从一个客户出发，查看风险、证据与模拟审核标记。',`<input id="customerId" class="input" type="number" value="${state.customerId}" aria-label="客户 ID"><button class="primary" id="customerSearch">查询客户</button>`)+`<div id="customerBody"><div class="loading">正在读取客户…</div></div>`;
  $('customerSearch').onclick=()=>{state.customerId=Number($('customerId').value);loadCustomer(state.customerId);};
  $('customerId').onkeydown=e=>{if(e.key==='Enter')$('customerSearch').click();};
  await loadCustomer(state.customerId);
}
async function loadCustomer(id) {
  const box=$('customerBody'); box.innerHTML='<div class="loading">正在计算客户风险解释…</div>';
  try {
    const data=await api('/api/customers/'+id), c=data.customer, risk=data.risk_level+'风险', s=data.shap;
    const readable=s.contributions.filter(x=>businessFeatures.has(x.feature));
    const positive=readable.filter(x=>x.contribution>0).slice(0,3), negative=readable.filter(x=>x.contribution<0).slice(0,3);
    const factorList=items=>items.length?items.map(item=>`<div class="impact-row"><span>${h(featureLabel(item.feature))}</span><strong class="${item.contribution>0?'risk-up':'risk-down'}">${pp(item.contribution)}</strong></div>`).join(''):'<p class="muted">没有明显贡献</p>';
    const guidance=data.risk_level==='高'?'建议人工复核':data.risk_level==='中'?'建议补充核验':'建议常规核验';
    const summary=`在可解读字段中，${positive.length?positive.map(x=>featureLabel(x.feature)).join('、')+'推动模型风险上升':'没有明显的风险上升项'}；${negative.length?negative.map(x=>featureLabel(x.feature)).join('、')+'抵消部分模型风险':'没有明显的风险抵消项'}。匿名字段的贡献保留在完整明细中，不能推断其业务原因。`;
    const top=readable.slice(0,6), shown=new Set(top.map(x=>x.feature));
    const remainder=s.contributions.filter(x=>!shown.has(x.feature)).reduce((sum,x)=>sum+x.contribution,0);
    let running=s.base_probability;
    const steps=[...top.map(x=>({label:featureLabel(x.feature),value:x.contribution})),{label:'其他（含匿名）特征',value:remainder}].map(x=>{running+=x.value;return `<div class="waterfall-step"><span>${h(x.label)}</span><div class="waterfall-track"><i class="${x.value>=0?'up':'down'}" style="width:${Math.max(2,Math.min(100,Math.abs(x.value)*400))}%"></i></div><strong class="${x.value>=0?'risk-up':'risk-down'}">${pp(x.value)}</strong><small>→ ${pct(running,1)}</small></div>`;}).join('');
    box.innerHTML=`<div class="review-hero card"><div><div class="eyebrow">CUSTOMER #${id} · XGBOOST</div><h2>客户 #${id}</h2><div class="risk-headline">${riskTag(risk)} <span>预测违约概率 <strong>${pct(s.prediction,2)}</strong></span></div><p class="recommendation">${guidance}</p><p class="muted">模型提供风险提示，最终决定由人工结合完整信息作出。</p></div></div>
    <div class="grid-2">${card('推动当前预测的因素','XGBoost · 概率空间 SHAP · 仅展示可解读字段',`<h3 class="minor-title">推高模型风险</h3>${factorList(positive)}<h3 class="minor-title">抵消模型风险</h3>${factorList(negative)}<p class="note-small">${h(s.note)} 匿名字段的贡献见完整明细。</p>`)}${card('客户核心资料','来自天池脱敏数据',metric('贷款金额','¥'+n(c.loanAmnt))+metric('年收入','¥'+n(c.annualIncome))+metric('贷款利率',n(c.interestRate,2)+'%')+metric('债务收入比',n(c.dti,2))+metric('信用等级 / FICO 下限',h(c.grade)+' / '+n(c.ficoRangeLow)))}</div>
    ${card('风险解释路径','所有特征贡献合计后等于当前 XGBoost 预测概率',`<div class="waterfall-end">基础风险 <strong>${pct(s.base_probability,2)}</strong></div>${steps}<div class="waterfall-end final">最终预测 <strong>${pct(s.prediction,2)}</strong></div><details class="technical-details"><summary>查看全部特征贡献与计算方法</summary><p>${h(s.method)} · ${n(s.background_rows)} 条背景样本</p><div class="table-wrap"><table><tr><th>特征</th><th>客户值（编码后）</th><th>SHAP 概率贡献</th></tr>${s.contributions.map(x=>`<tr><td>${h(featureLabel(x.feature))}</td><td>${n(x.value,2)}</td><td>${pp(x.contribution)}</td></tr>`).join('')}</table></div></details>`)}
    <div class="grid-2">${card('本地风险摘要','只依据 SHAP 结果整理；单客户记录不发送在线模型',`<p class="summary-text">${h(summary)}</p><p class="note-small">LR 基准对同一客户的预测为 ${pct(s.lr_probability,2)}；模型差异不等于特征交互的单独证明。</p>`)}${card('模拟审核标记','仅当前页面展示，不保存、不执行真实审批',`<div class="decision-actions"><button data-decision="通过">标记通过</button><button data-decision="人工复核">标记人工复核</button><button data-decision="拒绝">标记拒绝</button></div><p id="decisionState" class="muted">${state.decision[id]?'当前模拟标记：'+h(state.decision[id]):'尚未标记'}</p>`)}</div>`;
    document.querySelectorAll('[data-decision]').forEach(button=>button.onclick=()=>{state.decision[id]=button.dataset.decision;$('decisionState').textContent=`当前模拟标记：${button.dataset.decision}（仅本页有效）`;});
  } catch(error) { box.innerHTML=`<div class="error">${h(error.message)}</div>`; }
}

async function strategy() {
  const initial=await api('/api/threshold?value='+state.strategyThreshold);
  $('content').innerHTML=heading('POLICY SIMULATOR','策略模拟','调整审核阈值和成本假设，观察对客户与业务的影响。')+
  `<div class="card"><div class="form-grid"><div><label>需人工审核的风险阈值 <strong id="thresholdLabel">${n(state.strategyThreshold,2)}</strong></label><input class="slider" id="thresholdSlider" type="range" min="0.05" max="0.90" step="0.01" value="${state.strategyThreshold}"></div><div><label>漏掉一名违约客户的假设成本（元）</label><input class="input" id="badCost" type="number" min="0" value="20000"></div><div><label>额外审核一名正常客户的假设成本（元）</label><input class="input" id="goodCost" type="number" min="0" value="800"></div></div><p class="muted">以下是 16 万条有标签留出验证数据上的策略模拟，不代表未来真实损失。</p></div><div id="strategyResult"></div>`;
  const render=x=>{
    $('thresholdLabel').textContent=n(x.threshold,2);
    $('strategyResult').innerHTML=`<div class="kpi-grid business-kpis">${[['预计需人工审核',n(x.flagged),'验证集客户数'],['可识别违约客户',pct(x.recall),'历史违约样本的识别率'],['正常客户额外审核率',pct(x.false_reject_rate),'被标记的正常客户比例'],['假设策略成本','¥'+n(x.estimated_cost),'基于输入的成本假设']].map(([label,value,foot])=>`<div class="card kpi"><div class="label">${label}</div><span class="value">${value}</span><div class="foot">${foot}</div></div>`).join('')}</div>
    <div class="grid-2">${card('业务后果','阈值改变会同时影响漏判与审核工作量',metric('漏掉的违约客户',n(x.fn))+metric('额外审核的正常客户',n(x.fp))+metric('已识别的违约客户',n(x.tp)))}${card('如何计算','使用相同留出验证集比较不同阈值',`<div class="formula">假设策略成本 = 漏判 × ¥${n(x.bad_cost)} + 额外审核正常客户 × ¥${n(x.good_cost)}</div><p class="muted">真实策略还需结合授信金额、回收率、收益、法规与人工复核能力。</p><details class="technical-details"><summary>查看技术指标 TP / FP / FN / TN</summary>${metric('TP',n(x.tp))+metric('FP',n(x.fp))+metric('FN',n(x.fn))+metric('TN',n(x.tn))+metric('Precision',pct(x.precision))+metric('Recall',pct(x.recall))}</details>`)}</div>`;
  };
  render(initial);
  let timer;
  ['thresholdSlider','badCost','goodCost'].forEach(id=>$(id).oninput=()=>{clearTimeout(timer);state.strategyThreshold=Number($('thresholdSlider').value);timer=setTimeout(async()=>{try{render(await api(`/api/threshold?value=${state.strategyThreshold}&bad_cost=${$('badCost').value}&good_cost=${$('goodCost').value}`));}catch(error){$('strategyResult').innerHTML=`<div class="error">${h(error.message)}</div>`;}},130);});
}

async function analyst() {
  $('content').innerHTML=heading('AI RISK ANALYST','AI Analyst','自然语言提问 → 本地工具计算 → 证据 → DeepSeek 总结与操作入口。')+
  `<div class="grid-2 wide-left"><section class="card analyst-workspace"><div class="suggestions"><button data-question="为什么客户800001有风险？">解释客户 #800001</button><button data-question="为什么 D 级客户整体风险高？">D 级 SHAP 贡献</button><button data-question="为什么高风险客户这么多？">解释高风险客群</button><button data-question="把 D/E 级且 PD>0.6 的客户筛出来。">筛选 D/E 级客户</button><button data-question="如果审核阈值从0.35调到0.45，会有什么变化？">比较审核阈值</button></div><div id="analystAnswer" class="analyst-answer"><div class="empty">选择示例问题，或输入你关心的风险问题。</div></div><form id="analystForm" class="analyst-form"><input id="analystQuestion" maxlength="500" placeholder="例如：为什么客户 800001 有风险？" required><button class="primary" type="submit">分析</button></form></section>${card('工作方式','每条回答都能追溯到计算口径',`<div class="workflow-step">1 <strong>识别问题</strong><span>客群、策略、客户或模型健康</span></div><div class="workflow-step">2 <strong>调用本地工具</strong><span>预测、SHAP、客群或阈值计算</span></div><div class="workflow-step">3 <strong>展示证据与下一步</strong><span>客群聚合交给在线模型总结；单客户在本地回答</span></div><div class="notice">SHAP 是模型贡献，不是违约原因。单客户记录及解释不发送给 DeepSeek。测试集无真实违约标签。</div>`)}</div>`;
  document.querySelectorAll('[data-question]').forEach(button=>button.onclick=()=>{$('analystQuestion').value=button.dataset.question;submitAnalyst();});
  $('analystForm').onsubmit=e=>{e.preventDefault();submitAnalyst();};
}
async function submitAnalyst() {
  const question=$('analystQuestion').value.trim(); if (!question) return;
  $('analystAnswer').innerHTML='<div class="loading">正在查询数据证据并整理回答…</div>';
  try {
    const result=await api('/api/analyst',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question})});
    $('analystAnswer').innerHTML=`<div class="answer-bubble">${h(result.answer)}</div><div class="analyst-actions">${(result.actions||[]).map((a,i)=>`<button class="secondary" data-action="${i}">${h(a.label)} →</button>`).join('')}</div>${result.matches?.length?`<div class="table-wrap"><table><tr><th>客户 ID</th><th>等级</th><th>预测概率</th></tr>${result.matches.map(x=>`<tr class="clickable" data-match="${x.id}"><td>#${x.id}</td><td>${h(x.grade)}</td><td>${pct(x.probability,2)}</td></tr>`).join('')}</table></div>`:''}<div class="trace"><strong>计算路径：${h(result.route)}</strong><p>${h(result.calculation)}</p><p>回答模式：${h(result.mode)}</p><details><summary>查看计算证据</summary><pre>${h(JSON.stringify(result.evidence,null,2))}</pre></details></div>`;
    document.querySelectorAll('[data-action]').forEach(button=>button.onclick=()=>{const action=result.actions[Number(button.dataset.action)];if(action.customer_id)state.customerId=action.customer_id;if(action.threshold!=null)state.strategyThreshold=action.threshold;if(action.grade)state.filters.grade=action.grade;navigate(action.page);});
    document.querySelectorAll('[data-match]').forEach(row=>row.onclick=()=>{state.customerId=Number(row.dataset.match);navigate('customer');});
  } catch(error) { $('analystAnswer').innerHTML=`<div class="error">${h(error.message)}</div>`; }
}

async function data() {
  const dataset=await api('/api/overview'), response=await api('/api/columns');
  const qualityBody=`<details class="technical-details"><summary>查看缺失、重复与字段类型</summary>${metric('缺失单元格',n(dataset.missing_cells))}${metric('重复整行',n(dataset.duplicate_rows))}<input id="columnSearch" class="input search" placeholder="搜索字段…"><div class="table-wrap" style="max-height:400px;overflow:auto"><table><thead><tr><th>字段</th><th>类型</th><th>缺失率</th><th>唯一值</th></tr></thead><tbody id="columnRows"></tbody></table></div><div id="columnDetail"></div></details>`;
  $('content').innerHTML=heading('ADVANCED / DATA','数据管理','当前数据集与质量详情。',`<button class="secondary" id="backOverview">返回业务总览</button>`)+
  `<div class="grid-3">${[['当前数据集','Tianchi Credit 531830'],['训练数据',n(dataset.train_rows)+' 条'],['待评估数据',n(dataset.test_rows)+' 条']].map(([label,value])=>`<div class="card kpi"><div class="label">${label}</div><span class="value" style="font-size:20px">${value}</span></div>`).join('')}</div>`+
  card('数据质量','打开技术详情时再查看字段指标',qualityBody)+
  card('上传文件预览','CSV / Parquet · 最多 25 MB · 不保存、不替换当前数据集',`<div class="control-row"><input id="uploadFile" type="file" accept=".csv,.parquet"><button class="secondary" id="uploadButton">预览数据质量</button></div><div id="uploadResult"></div>`);
  $('backOverview').onclick=()=>navigate('overview');
  const draw=filter=>{
    $('columnRows').innerHTML=response.columns.filter(x=>x.name.toLowerCase().includes(filter.toLowerCase())).map(x=>`<tr class="clickable" data-column="${h(x.name)}"><td>${h(x.name)}</td><td>${h(x.type)}</td><td>${n(x.missing_pct,2)}%</td><td>${n(x.unique)}</td></tr>`).join('');
    document.querySelectorAll('[data-column]').forEach(row=>row.onclick=async()=>{
      const x=await api('/api/columns/'+encodeURIComponent(row.dataset.column));
      const distribution=x.distribution.map(item=>`<tr><td>${h(item.label)}</td><td>${n(item.good)}</td><td>${n(item.bad)}</td><td>${pct(item.bad/(item.good+item.bad||1))}</td></tr>`).join('');
      $('columnDetail').innerHTML=`<h3>${h(x.column.name)}</h3>${x.column.median==null?'':metric('P1 / 中位数 / P99',n(x.column.p01,2)+' / '+n(x.column.median,2)+' / '+n(x.column.p99,2))}${distribution?`<div class="table-wrap"><table><tr><th>分组</th><th>正常</th><th>违约</th><th>违约率</th></tr>${distribution}</table></div>`:'<p class="muted">该字段未预计算分布。</p>'}`;
    });
  };draw('');$('columnSearch').oninput=e=>draw(e.target.value);
  $('uploadButton').onclick=async()=>{const file=$('uploadFile').files[0];if(!file)return;const form=new FormData();form.append('file',file);$('uploadResult').textContent='解析中…';try{const x=await api('/api/upload-preview',{method:'POST',body:form});$('uploadResult').innerHTML=`<p class="muted">${h(x.filename)} · ${n(x.preview_rows)} 行 · ${h(x.note)}</p><div class="table-wrap"><table><tr><th>字段</th><th>类型</th><th>缺失率</th></tr>${x.columns.map(c=>`<tr><td>${h(c.name)}</td><td>${h(c.type)}</td><td>${n(c.missing_pct,1)}%</td></tr>`).join('')}</table></div>`;}catch(error){$('uploadResult').innerHTML=`<div class="error">${h(error.message)}</div>`;}};
}

async function features() {
  const [features,global]=await Promise.all([api('/api/features'),api('/api/shap/global')]);
  const active=new Set(global.feature_importance.map(x=>x.feature));
  const mapped={loan_income_ratio:'loan_to_income',credit_history_years:'credit_history_months'};
  const top=global.feature_importance.slice(0,10).map(x=>`<tr><td>${h(featureLabel(x.feature))}<small class="muted"> · ${h(x.feature)}</small></td><td>${n(x.mean_abs*100,2)} pp</td><td>${x.mean_signed>.002?'整体偏向推高':x.mean_signed<-.002?'整体偏向降低':'正负混合'}</td></tr>`).join('');
  $('content').innerHTML=heading('ADVANCED / FEATURES','模型风险洞察','先看 SHAP 模型贡献，再查看特征公式与样本统计。')+
  card('全局特征洞察','测试集 A 固定抽样 · mean(|SHAP|)；方向是样本平均贡献，不代表因果',`<div class="table-wrap"><table><tr><th>特征</th><th>平均绝对贡献</th><th>抽样平均方向</th></tr>${top}</table></div>`)+
  card('原始字段与派生特征','点击派生特征查看公式与样本统计',`<div class="table-wrap"><table><thead><tr><th>特征</th><th>业务含义</th><th>来源</th><th>状态</th></tr></thead><tbody>${[['interestRate','贷款利率','Raw'],['dti','债务收入比','Raw'],['ficoRangeLow','FICO 下限','Raw'],...Object.entries(features).map(([key,value])=>[key,value.label,'Derived'])].map(([key,label,source])=>`<tr class="clickable" data-feature="${h(key)}"><td><strong>${h(key)}</strong></td><td>${h(label)}</td><td>${source}</td><td>${tag(active.has(mapped[key]||key)?'当前模型':'候选')}</td></tr>`).join('')}</tbody></table></div><div id="featureDetail" class="empty">选择特征查看详情</div>`);
  document.querySelectorAll('[data-feature]').forEach(row=>row.onclick=async()=>{const key=row.dataset.feature, item=features[key];if(!item){$('featureDetail').innerHTML=`<h3>${h(key)}</h3><p class="muted">原始字段来自天池数据集。</p>`;return;}$('featureDetail').innerHTML='<div class="loading">计算样本预览…</div>';try{const x=await api('/api/features/'+key+'/preview');$('featureDetail').innerHTML=`<h3>${h(item.label)}</h3><div class="formula">${h(key)} = ${h(item.formula)}</div><p class="muted">${h(item.description)} · 预览前 ${n(x.sample_rows)} 条训练记录</p>${metric('缺失率',n(x.missing_pct,2)+'%')}${metric('中位数 / P99',n(x.median,2)+' / '+n(x.p99,2))}`;}catch(error){$('featureDetail').innerHTML=`<div class="error">${h(error.message)}</div>`;}});
}

async function models() {
  const [data,conf,global]=await Promise.all([api('/api/models'),api('/api/threshold'),api('/api/shap/global')]);
  const baseline=data.baseline, importance=global.feature_importance.slice(0,10), maximum=importance[0]?.mean_abs||1;
  const comparison=`<div class="table-wrap"><table><thead><tr><th>模型</th><th>用途</th><th>AUC</th><th>KS</th><th>PR-AUC</th><th>Brier ↓</th></tr></thead><tbody>${data.comparison.map(x=>`<tr><td><strong>${h(x.algorithm)}</strong></td><td>${h(x.purpose)}</td><td>${n(x.auc,4)}</td><td>${n(x.ks,4)}</td><td>${n(x.pr_auc,4)}</td><td>${n(x.brier,4)}</td></tr>`).join('')}</tbody></table></div><p class="note-small">两模型使用同一训练划分与 16 万条留出验证。Brier 越低越好；概率可靠性还应查看校准曲线。</p>`;
  const bars=importance.map(x=>`<div class="bar-row shap-importance"><span title="${h(x.feature)}">${h(featureLabel(x.feature))}</span><div class="bar-track"><div class="bar-fill" style="width:${x.mean_abs/maximum*100}%"></div></div><span class="bar-value">${n(x.mean_abs*100,2)} pp</span></div>`).join('');
  $('content').innerHTML=heading('ADVANCED / MODEL','模型表现与风险洞察','同一留出集比较 LR 与 XGBoost；SHAP 解释当前主模型。')+
  card('基准与主模型','Logistic Regression 提供传统透明基准；XGBoost 用于当前风险预测',comparison)+
  `<div class="kpi-grid">${[['XGBoost AUC',n(baseline.auc,4)],['KS',n(baseline.ks,4)],['PR-AUC',n(baseline.pr_auc,4)],['Brier ↓',n(baseline.brier,4)]].map(([label,value])=>`<div class="card kpi"><div class="label">${label}</div><span class="value">${value}</span></div>`).join('')}</div>`+
  `<div class="grid-2">${card('全局模型风险贡献','测试集 A 抽样 · mean(|SHAP|) · 单位：百分点',bars)}${card('SHAP 散点分布','展示特征值与模型概率贡献的方向',beeswarmChart(global))}</div>`+
  `<p class="note-small">${h(global.note)} 背景样本 ${n(global.background_rows)} 条，解释样本 ${n(global.sample_rows)} 条。</p>`+
  `<div class="grid-2">${card('校准曲线','预测概率 vs 实际违约比例；对角线表示理想校准',curve(baseline.calibration,'预测概率','实际比例','#5385b5'))}${card('验证集混淆矩阵','审核阈值 0.35',metric('识别违约 TP',n(conf.tp))+metric('误标正常 FP',n(conf.fp))+metric('漏掉违约 FN',n(conf.fn))+metric('放行正常 TN',n(conf.tn)))}</div>`+
  `<details class="technical-details"><summary>查看 ROC 与 PR 曲线</summary><div class="grid-2">${card('ROC 曲线','留出验证集',curve(baseline.roc))}${card('PR 曲线','留出验证集',curve(baseline.pr,'召回率','精确率','#e49b44'))}</div></details>`+
  card('其他算法小样本实验','仅供技术探索；样本规模不同，不能与上方全量比较',`<div class="control-row"><select id="algorithm"><option value="hist_gradient_boosting">梯度提升树</option><option value="logistic">逻辑回归</option><option value="random_forest">随机森林</option></select><input id="sampleRows" class="input" type="number" min="5000" max="50000" step="5000" value="30000" aria-label="训练样本数"><button id="trainButton" class="primary">开始训练</button></div><p id="trainStatus" class="muted">结果仅保存在当前服务进程。</p><div class="table-wrap"><table><thead><tr><th>算法</th><th>样本</th><th>AUC</th><th>KS</th><th>PR-AUC</th></tr></thead><tbody id="experimentRows"></tbody></table></div>`);
  const draw=rows=>{$('experimentRows').innerHTML=rows.map(x=>`<tr><td>${h(x.algorithm)}</td><td>${n(x.sample_rows)}</td><td>${n(x.auc,4)}</td><td>${n(x.ks,4)}</td><td>${n(x.pr_auc,4)}</td></tr>`).join('')||'<tr><td colspan="5">尚无小样本实验</td></tr>';};draw(data.experiments);
  $('trainButton').onclick=async()=>{const button=$('trainButton');button.disabled=true;$('trainStatus').textContent='训练中…';try{const job=await api('/api/training/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({algorithm:$('algorithm').value,sample_rows:Number($('sampleRows').value)})});const poll=async()=>{const x=await api('/api/training/jobs/'+job.id);if(x.status==='running'){setTimeout(poll,1200);return;}button.disabled=false;if(x.status==='failed'){$('trainStatus').textContent='失败：'+x.error;return;}$('trainStatus').textContent=`已完成 · AUC ${n(x.auc,4)}`;draw((await api('/api/models')).experiments);};poll();}catch(error){button.disabled=false;$('trainStatus').textContent=error.message;}};
}

async function monitor() {
  const data=await api('/api/monitor');
  $('content').innerHTML=heading('ADVANCED / HEALTH','模型健康','训练集与无标签测试集 A 的输入分布对照。')+
  `<div class="notice">${h(data.note)}</div>`+
  card('特征漂移 PSI','< 0.10 正常；0.10–0.25 关注；≥ 0.25 漂移',`<div class="table-wrap"><table><thead><tr><th>特征</th><th>PSI</th><th>状态</th></tr></thead><tbody>${data.drift.map(x=>`<tr><td>${h(x.name)}</td><td>${n(x.psi,4)}</td><td>${tag(x.status)}</td></tr>`).join('')}</tbody></table></div><p class="note-small">PSI 说明输入分布变化，不能单独证明模型效果下降；测试集无标签，不能计算线上 AUC。</p>`);
}

navigate('overview');
