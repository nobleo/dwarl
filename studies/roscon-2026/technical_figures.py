"""Editable technical figures for the ROSCon deck (python-pptx).

Dimensions below describe the frozen iteration-200 bundle, not toolkit defaults.
"""

# Figure copy remains on one source line per text box for editorial review.
# ruff: noqa: E501

from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.util import Inches, Pt


def draw(slide, name, x=.9, y=2.4):
    ink, blue, teal, gray = '14324F', '4285F4', '0090A0', '4A637D'

    def text(px, py, w, h, value, size=19, color=ink, bold=False):
        box = slide.shapes.add_textbox(Inches(x+px), Inches(y+py), Inches(w), Inches(h))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = 0
        for i, line in enumerate(value.split('\n')):
            p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            p.text = line
            p.font.name = 'Lato'
            p.font.size = Pt(size)
            p.font.bold = bold
            p.font.color.rgb = RGBColor.from_string(color)
        return box

    def box(px, py, w, h, title, body, color=blue):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                      Inches(x+px), Inches(y+py), Inches(w), Inches(h))
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string('F4F8FC')
        shape.line.color.rgb = RGBColor.from_string(color)
        text(px+.2, py+.12, w-.4, .5, title, 21, color, True)
        text(px+.2, py+.72, w-.4, h-.75, body)

    def arrow(ax, ay, bx, by):
        line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
                                         Inches(x+ax), Inches(y+ay), Inches(x+bx), Inches(y+by))
        line.line.color.rgb = RGBColor.from_string(gray)
        line.line.width = Pt(2)
        from pptx.oxml.xmlchemy import OxmlElement
        end = OxmlElement('a:tailEnd')
        end.set('type', 'triangle')
        line.line._get_or_add_ln().append(end)

    if name == 'Nav2Architecture':
        box(0, 0, 4.1, 1.8, 'Goal + behavior tree', 'NavigateToPose\nPlan / follow / recover')
        box(5, 0, 4.1, 1.8, 'NavFn global planner', 'Map + robot pose\nGlobal path in map frame')
        arrow(4.1, .9, 5, .9)
        box(0, 2.7, 4.1, 2.1, 'Observation builder', 'TF + odometry + scan\nPath lookaheads in robot frame\nLidar history and motion state', teal)
        box(5, 2.7, 4.1, 2.1, 'DWARL controller plugin', 'ONNX Runtime → 2 actions\nVelocity-window mapping\n10 Hz / 100 ms period', teal)
        arrow(7.05, 1.8, 7.05, 2.7)
        arrow(4.1, 3.75, 5, 3.75)
        box(10, 2.7, 4.1, 2.1, 'Collision monitor', 'Scan-based stop / approach\nIndependent command guard\nNot a learned safety guarantee')
        arrow(9.1, 3.75, 10, 3.75)
        box(10, 5.8, 4.1, 1.8, 'Robot / simulator', 'TwistStamped → drive\nOdometry + scan feedback')
        arrow(12.05, 4.8, 12.05, 5.8)
        box(0, 5.8, 8.9, 1.8, 'Recovery remains a system-level choice',
            'Progress checker / controller failure trigger the behavior tree.\nSelected experiments route backup through the same monitor.')
    elif name == 'PolicyLearning':
        box(0, 0, 6.5, 2, 'Lidar input · [1, 1, 3, 216]',
            '216 rays × history offsets {0, 2, 5} control ticks\nLog-normalized ranges; full mounting transform\nFront coverage: no inferred rear visibility', teal)
        box(7.3, 0, 6.8, 2, 'State input · [1, 16]',
            '4 path points × {distance, sin bearing, cos bearing}\nPrevious 2 clipped actions + normalized v and ω\nLookahead distances: 0.5 / 1 / 2 / 3 m', teal)
        box(3.7, 2.8, 6.7, 1.5, 'Exported actor · normalization in graph',
            'Lidar encoding + state → deterministic action pair')
        arrow(3.25, 2, 5.4, 2.8)
        arrow(10.7, 2, 8.8, 2.8)
        box(0, 5.1, 6.5, 2.5, 'Action mapping · [1, 2]',
            'Clip policy outputs to [−1, 1]\nMap into acceleration-feasible velocity window\nv: ±0.5 m/s; ω: ±1 rad/s\nLimits: 1 m/s² and 2 rad/s²')
        arrow(5.4, 4.3, 3.25, 5.1)
        box(7.3, 5.1, 6.8, 2.5, 'Training only · PPO + rewards',
            'Progress, clearance, tracking and terminal outcomes\nCritic uses training information; actor is exported\nReward changes require a new training run\nSafety monitor is outside the policy', teal)
    elif name == 'ImprovementLoop':
        box(0, 0, 6.5, 1.8, '1 · Freeze a validation experiment',
            'Checkpoint / robot contract / source revision\nCase IDs, stack, scoring and bounded trial budget')
        box(7.3, 0, 6.8, 1.8, '2 · Read individual evidence',
            'summary.json → episodes.json → trace / compute\nRequested vs measured motion; terminal pose; logs')
        arrow(6.5, .9, 7.3, .9)
        box(7.3, 2.7, 6.8, 2, '3 · Form one testable intervention',
            'Controller parameters, safety/recovery or reward\nDeclare success gain + collision guardrail\nRecord rejected variants and inactive counters', teal)
        arrow(10.7, 1.8, 10.7, 2.7)
        box(0, 2.7, 6.5, 2, '4 · Repeat matched validation',
            'Audit workload, snapshots and terminal consistency\nPromote only repeatable gains; preserve parent\nStop on no-improvement patience or trial budget', teal)
        arrow(7.3, 3.7, 6.5, 3.7)
        box(0, 5.7, 14.1, 1.9, '5 · Freeze selection → held-out evaluation → publish evidence',
            'Final test cases never choose the settings. Run Gazebo sequentially; isolate compute measurements.\nAgent edits happen between experiments; the agent is never part of the robot control loop.')
        arrow(3.25, 4.7, 3.25, 5.7)
    elif name == 'DeployPipeline':
        box(0, 0, 6.5, 2, 'Training machine',
            'Robot assets + frozen contract + dataset identity\nPPO training → completed checkpoint manifest\nSelect by validation, not by latest filename')
        box(7.3, 0, 6.8, 2, 'Export and parity checks',
            'Exact checkpoint hash → deterministic ONNX actor\nCompare runtime observations/actions with training\nPreserve lidar transform and action semantics', teal)
        arrow(6.5, 1, 7.3, 1)
        box(7.3, 2.8, 6.8, 2, 'Immutable deployment bundle',
            'policy.onnx + deploy.yaml + robot identity\nInput shapes, normalization, history, frames\nVelocity / acceleration limits and control period', teal)
        arrow(10.7, 2, 10.7, 2.8)
        box(0, 2.8, 6.5, 2, 'C++ Nav2 plugin + ONNX Runtime',
            'Build observations → inference → velocity window\nFollowPath output → independent collision monitor\nTraining libraries are absent from this control path')
        arrow(7.3, 3.8, 6.5, 3.8)
        box(0, 5.7, 14.1, 1.9, 'Measure at the correct boundary',
            'Recorded-input ONNX replay: model inference on the measured computer.\nController-server CPU/RSS: plugin + local costmap + ROS. Gazebo is separate; neither proves robot timing.')
        arrow(3.25, 4.8, 3.25, 5.7)
    else:
        raise ValueError(name)
