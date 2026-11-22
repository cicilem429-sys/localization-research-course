"""Build the PDF and SVG figures from saved experiment results; no model calls."""
import json
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Circle
from reportlab.graphics import renderSVG

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'final_validation_results.json').read_text())
FONT = Path('/Library/Fonts/Arial Unicode.ttf')
if not FONT.exists():
    import os
    FONT = Path(os.environ['REPORT_FONT'])
pdfmetrics.registerFont(TTFont('CJK', str(FONT)))
C = ['#64748b', '#38a2b8', '#0f766e', '#d89623', '#8b68b9']
METHODS = list(DATA['summary'])
NAMES = ['Huber', 'Fixed geometry', 'Adaptive geometry', 'V5 weights', 'V6 support']

def bars():
    d=Drawing(470,235)
    d.add(String(8,219,'Mean macro-CEP90 across 5 seeds (m)',fontName='Helvetica',fontSize=12))
    for j in range(5):
        x=108+j*72
        d.add(Line(x,39,x,198,strokeColor=colors.HexColor('#e2e8f0')))
        d.add(String(x-6,24,f'{j*.2:.1f}',fontSize=9))
    for i,(m,label) in enumerate(zip(METHODS,NAMES)):
        y=180-i*31; s=DATA['summary'][m]; v=s['macro_cep90_mean']; sd=s['macro_cep90_std']
        d.add(String(2,y+1,label,fontSize=9))
        d.add(Rect(108,y-5,v*360,16,fillColor=colors.HexColor(C[i]),strokeColor=None))
        x=108+v*360
        d.add(Line(x-sd*360,y+3,x+sd*360,y+3,strokeColor=colors.black))
        for xx in [x-sd*360,x+sd*360]:d.add(Line(xx,y-1,xx,y+7))
        d.add(String(x+sd*360+5,y,f'{v:.4f}',fontSize=9))
    d.add(String(108,5,'Error bars: sample standard deviation across seeds (not CI)',fontSize=8))
    return d

def seeds():
    d=Drawing(470,245)
    d.add(String(8,228,'Macro-CEP90 for each held-out seed (m)',fontSize=12))
    for k in range(5):
        y=45+k*35
        d.add(Line(45,y,444,y,strokeColor=colors.HexColor('#e2e8f0')))
        d.add(String(10,y-3,f'{.5+k*.1:.1f}',fontSize=9))
    for j,seed in enumerate(DATA['seeds']):d.add(String(35+j*100,28,str(seed),fontSize=9))
    for i,m in enumerate(METHODS):
        vals=DATA['summary'][m]['macro_cep90_per_seed']; points=[(45+j*100,45+(v-.5)*350) for j,v in enumerate(vals)]
        for a,b in zip(points,points[1:]):d.add(Line(*a,*b,strokeColor=colors.HexColor(C[i]),strokeWidth=1.5))
        for x,y in points:d.add(Circle(x,y,2.8,fillColor=colors.HexColor(C[i]),strokeColor=None))
        d.add(String(8+i*92,7,NAMES[i],fontSize=7.5,fillColor=colors.HexColor(C[i])))
    return d

for n,fn in [('method_comparison',bars),('per_seed',seeds)]:
    renderSVG.drawToFile(fn(),str(ROOT/'figures'/f'{n}.svg'))

styles={
 'title':ParagraphStyle('title',fontName='CJK',fontSize=23,leading=34,textColor=colors.HexColor('#113b51'),spaceAfter=20),
 'sub':ParagraphStyle('sub',fontName='CJK',fontSize=12,leading=20,textColor=colors.HexColor('#4c6470'),spaceAfter=18),
 'h':ParagraphStyle('h',fontName='CJK',fontSize=16,leading=24,textColor=colors.HexColor('#0f766e'),spaceAfter=14),
 'p':ParagraphStyle('p',fontName='CJK',fontSize=10.5,leading=18,spaceAfter=10),
 'small':ParagraphStyle('small',fontName='CJK',fontSize=8.5,leading=14,spaceAfter=8),
 'formula':ParagraphStyle('formula',fontName='CJK',fontSize=11,leading=20,backColor=colors.HexColor('#edf5f6'),borderPadding=8,spaceBefore=7,spaceAfter=15),
}
story=[]
def safe(text):
    return escape(str(text).replace('ᵢ', 'i').replace('ⱼ', 'j').replace('ₓ', 'x').replace('ᵀ', '^T')).replace('\n','<br/>')
def p(text,kind='p'):story.append(Paragraph(safe(text),styles[kind]))
def page(title):
 if story:story.append(PageBreak())
 p(title,'h')
def table(rows,widths):
 t=Table([[Paragraph(safe(c),styles['small']) for c in row] for row in rows],colWidths=widths,hAlign='LEFT')
 t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4f0f2')),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#c4d3d7')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),4)]));story.append(t);story.append(Spacer(1,12))

p('众包鲁棒定位算法的\n智能体探索与实验验证','title')
p('《人工智能与数学》课程研究报告\n依据实际 V1–V6 源码与保存实验记录整理 | 2026-11-22','sub')
p('摘要','h')
p('本文研究二维静态目标的带异常测距定位问题，将大语言模型的受限工具调用用于候选参数和权重构造的实验探索。源码保留了 V1–V6 的研究阶段，最终在五个随机种子生成的 600 组合成观测上，对 Huber、固定几何、自适应几何、V5 连续加权与 V6 多候选支持度进行固定参数比较。')
p('保存结果显示，自适应几何的平均 macro-CEP90 为 0.5715 m，五个种子均优于其他四种方法。V5 与 V6 分别为 0.5977 m 和 0.6214 m，未超过最佳基线。该结果说明，增加权重结构的复杂度未在当前实验中带来稳定收益。')
p('研究贡献是形成可核查的“假设—数值工具—反馈—修正”实验过程，并报告失败与适用边界，而非宣称发现全新最优定位算法。本文依据实际文件编制；结构化历史不是完整 API 对话，合成数据不代表真实 HarmonyOS 设备测试。')
p('关键词：非线性定位；Huber 损失；几何一致性；MAD；数学研究智能体','small')
p('材料依据：根目录 Python 源码、V2–V6 history JSON、final_validation_results.json、final_validation_summary.csv；来源哈希见 docs/source_manifest.json。','small')

page('1 研究问题与数学模型')
p('1.1 问题范围','sub')
p('课程选题背景为众包离线高精度位置聚合。本项目将其简化为二维静态、已知节点坐标和直接距离观测的子问题，不建模 RSSI 到距离的信号传播，也不处理运动状态或设备同步。')
p('设目标 x∈R²，节点 aᵢ∈R²，观测 dᵢ=||x−aᵢ||₂+εᵢ+bᵢ。其中 εᵢ 为随机测距噪声，bᵢ 为少量观测的正向异常扰动。估计阶段只能访问节点和观测距离，真实位置用于实验评价。')
p('rᵢ(x)=||x−aᵢ||₂−dᵢ；最小二乘：minₓ Σᵢ rᵢ(x)²。','formula')
p('异常观测的平方残差可能主导目标，因此采用 Huber 鲁棒目标。非线性问题仍可能存在局部极值，多初值求解不构成全局最优保证。')
p('ρδ(r)=r²/2（|r|≤δ）；ρδ(r)=δ(|r|−δ/2)（|r|>δ）。加权目标：minₓ Σᵢ wᵢρδ(rᵢ(x))。','formula')
p('1.2 几何一致性候选','sub')
p('对三节点取平方测距方程的差，可得线性系统：2(aⱼ−aᵢ)ᵀx=dᵢ²−dⱼ²+||aⱼ||²−||aᵢ||²。枚举三元组并跳过病态组合，得到候选位置，再以全部观测残差判定一致集合。')
p('fast_geometry.py 选择一致节点数最多的候选，平局时比较一致节点残差平方和；一致节点至少三个时，重新做非线性最小二乘。固定阈值方法使用 1.5 m。该方法是最大一致性思想的实现，不在本文主张数学原创性。')
p('数学实现依据：localization_tools.py、fast_geometry.py、weighted_localization.py。','small')

page('2 自适应阈值与连续权重')
p('2.1 自适应几何基线','sub')
p('先以 δ=0.3 的 Huber 估计初始位置，计算有符号残差，以中位数绝对偏差估计尺度。最终比较使用 c=2.0、最小阈值 0.5 m。')
p('σ̂=1.4826 medianᵢ |rᵢ−medianⱼ(rⱼ)|；T=max(0.5,2σ̂)。','formula')
p('随后用阈值 T 运行几何一致性筛选与重估计。尺度常数来自既有稳健估计形式；小样本、拟合残差的相关性和异常比例会影响其校准，本文不提供误差界证明。')
p('2.2 V5：高斯连续权重','sub')
p('从固定几何解计算残差，最终冻结参数 τ=4.0、δ=1.0、最小权重 0。以 wᵢ=exp[−0.5(|rᵢ|/τ)²] 做加权 Huber 优化，检验“保留弱可信观测是否优于硬删除”的假设。')
p('2.3 V6：多候选支持度','sub')
p('对每个候选位置，计算所有节点是否满足残差阈值。评分时排除构造该候选的三个节点，避免直接自我投票。选取得分最高的 top-k 个候选，以 softmax 质量权重汇总各节点的支持比例 sᵢ，再映射为可信权重。')
p('wᵢ=wmin+(1−wmin)sᵢ^γ。冻结参数：阈值 0.5，top-k=120，γ=0.3，wmin=0.1，δ=0.7，temperature=1.0。','formula')
p('若某节点没有独立投票分母，原实现回退为所有选中候选的支持。因此这是一项工程启发式，不是严格独立的概率估计。V6 以自适应几何位置作加权求解初值。BFGS 只要输出有限值就可能纳入评价，即使 success=False；该求解器局限应纳入后续审计。')
p('实现依据：adaptive_ten_anchor.py；research_agent_v6.py 中 evaluate_v5、geometry_support_weights 与 evaluate_candidate。','small')

page('3 智能体结构与 V1–V6 迭代')
p('控制流程为：向模型提供研究目标与已完成实验摘要；模型通过受限函数工具提出参数和假设；Python 工具验证参数并执行数值求解；把评价反馈给模型并记录下一轮。候选方法族、参数边界与数据环境由代码规定，不能视为无限制自主数学研究。')
table([['阶段','实际入口与记录','研究内容'],['V1','localization_agent.py\nlocalization_history.json','Huber 参数选择，4 次工具结果'],['V2','localization_agent_v2.py\n同名 history','Huber、固定/自适应几何比较，3 轮'],['V3','research_agent_v3.py\n同名 history','四场景统一环境与保留集，3 轮'],['V4','research_agent_v4.py\n同名 history','基于估计噪声的策略切换，4 轮'],['V5','research_agent_v5.py\n同名 history','连续权重函数探索，12 轮'],['V6','research_agent_v6.py\n同名 history','多候选支持度与权重，16 轮']],[43,245,190])
p('V4 保留集切换策略 macro-CEP90=0.6704，高于自适应基线 0.6332。V5、V6 的开发集选择也不能直接等同于对新数据的改进，因此最终另以五个固定新种子验证。不同版本配置不同，不画跨版本分数递降曲线。')
p('V5 历史记录报告 12 次请求、38689 输入 token、2222 输出 token，估计费用 0.449014 元；V6 报告 16 次请求、74692 输入 token、3890 输出 token，估计费用 0.854984 元。上述是本地记录和历史价格常数计算，不是经服务商账单核验的实际费用。')
p('记录包含候选假设、参数、指标和部分模型总结；未发现完整逐条 API 请求/响应轨迹，V1 仅有工具实验结果。本文不把保存摘要表述为完整模型审计证据。本次交付没有重新调用付费 API。','small')

page('4 最终实验设计与比较结果')
p('固定随机种子 11001、11002、11003、11004、11005。每个种子有四场景、三噪声水平（σ=0.2、0.5、1.0 m）、每格十次，共 120 组，五批共 600 组。每组十节点，目标均匀分布于 [2,8]²。随机布局节点均匀分布于 [−5,15]²，其余使用预置十节点坐标。')
p('四场景为：固定布局单个 +8 m 异常（standard）；随机布局单个 +8 m 异常；固定布局两个 +8 m 异常；固定布局单个 +2 m 异常。距离加入高斯噪声后截断为非负。standard 仍有异常，不是无粗差的干净场景。')
p('每种子五种方法使用同一批观测。误差为估计位置与真实位置的欧氏距离；macro-CEP90 为 12 个子组各自经验 90% 分位数的平均。每格只有十例，其尾部分位数具有较大不确定性。')
table([['方法','macro-CEP90/m','跨种子 SD/m','平均误差/m']]+[[label,f"{DATA['summary'][m]['macro_cep90_mean']:.4f}",f"{DATA['summary'][m]['macro_cep90_std']:.4f}",f"{DATA['summary'][m]['mean_error_average']:.4f}"] for m,label in zip(METHODS,NAMES)],[153,114,105,106])
story.append(bars())
p('图 1 由原始保存 JSON 直接生成。误差棒为跨种子样本标准差，不是 95% 置信区间。最终验证不调用大模型，也不再次调整候选参数。','small')

page('5 统计解释、失败与适用边界')
story.append(seeds());story.append(Spacer(1,12))
p('自适应几何在五个种子中全部获得最低 macro-CEP90，其跨种子均值比 Huber 低约 23.1%。这支持当前模拟设定中的基线选择，不证明对任意噪声、布局或设备的普遍优势。')
table([['配对差值：候选−Adaptive','均值/m','探索性 95% t 区间/m','胜/负/平'],['V5','+0.0263','约 [−0.0000, 0.0525]','0/5/0'],['V6','+0.0499','约 [0.0132, 0.0867]','0/5/0']],[163,70,175,70])
p('区间基于五个种子差值的均值、样本标准差和自由度 4 的 t 临界值。保存值经四舍五入，V5 下界接近零；不把 −0.0000 当作严格的零，也不据此给出显著性定论。V6 区间为正只是探索性证据；未做多重比较控制或大规模种子检验。')
p('失败分析：V5 的弱权重可能仍保留异常影响；V6 候选支持度受到阈值与候选噪声影响，增加复杂度可能引入估计不确定性。这里是待验证解释，尚无消融实验建立因果结论。')
p('现有实验比较的是算法结果，未统一完整求解预算或墙钟时间；没有证明模型研究比网格、随机或人工搜索更优。没有真实设备数据、动态目标或 RSSI 标定，不能宣称达到工业精度指标。最终 JSON 只保存聚合指标，没有全部逐样本误差档案。')

page('6 复现、材料核验与结论')
p('6.1 可复现交付','sub')
p('原始 Python/JSON/CSV 来自实际可访问的 my-math-agent；docs/source_manifest.json 记录复制时的 SHA-256。源码保留根目录布局以维持原有导入。只对 V6 做 API 客户端延迟初始化，对最终验证增加输出目录选项，数值算法与冻结参数不变。')
p('安装 requirements.txt 后，在仓库根目录执行：','p')
p('python final_statistical_validation.py --output-dir results/reproduced\npython verify_delivery.py','formula')
p('离线验证不需要 API 密钥。原始 JSON/CSV 保留在根目录，重跑文件写入独立目录。requirements 固定于实际原实验环境中读取的版本；跨平台数值优化可能有微小差异。报告和 SVG 可用 scripts/build_report.py 与 requirements-report.txt 重新生成。')
p('6.2 材料完整性','sub')
p('已取得 V1–V6 相关源码、V2–V6 结构化研究历史、最终五种子 JSON/CSV。原先没有最终 PDF，本报告由实际材料新编制。缺失的完整模型对话、账单、逐样本误差与真实设备观测没有伪造补充。未上传环境配置、API 密钥或虚拟环境。')
p('6.3 结论','sub')
p('本项目完成了受限数学工具驱动的研究智能体原型与鲁棒定位仿真实验链。在 600 组合成观测的固定参数验证中，自适应几何优于 V5 和 V6，说明复杂权重构造未必优于简单稳健筛选。主要成果是可追溯的实验探索与明确的负结果，而不是超过既有方法的精度宣称。后续应优先补齐逐样本档案、求解器收敛审计、同预算对照和真实设备验证。')
p('材料与实现索引','sub')
p('[1] research_environment_v3.py：数据配置。\n[2] localization_tools.py、fast_geometry.py、adaptive_ten_anchor.py：基线。\n[3] research_agent_v5.py、research_agent_v6.py、weighted_localization.py：候选。\n[4] *_history.json：结构化研究记录。\n[5] final_statistical_validation.py、final_validation_results.json、CSV：最终验证。\n[6] docs/PROVENANCE.md、source_manifest.json：来源与整理修改说明。','small')
p('代码与报告由 AI 协助整理；课程提交者应独立复现并能解释数学定义、程序行为及研究局限。','small')

def footer(c,doc):
 c.setStrokeColor(colors.HexColor('#c4d3d7'));c.line(45,41,550,41)
 c.setFont('CJK',8);c.setFillColor(colors.HexColor('#607684'));c.drawString(45,28,'人工智能与数学 | 鲁棒定位研究');c.drawRightString(550,28,str(doc.page))

SimpleDocTemplate(str(ROOT/'report/final_report.pdf'),pagesize=(595.28,841.89),rightMargin=50,leftMargin=45,topMargin=44,bottomMargin=56,title='众包鲁棒定位算法的智能体探索与实验验证',author='课程项目').build(story,onFirstPage=footer,onLaterPages=footer)
print('Built report/final_report.pdf and two SVG charts from saved JSON')
