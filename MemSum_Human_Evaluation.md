<a href="https://colab.research.google.com/github/nianlonggu/MemSum/blob/main/MemSum_Human_Evaluation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# MemSum Human Evaluation

## Install Dependencies


```python
!pip install -r requirements.txt --quiet
```

      Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m60.5/60.5 kB[0m [31m1.9 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.4/7.4 MB[0m [31m20.3 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m9.2/9.2 MB[0m [31m60.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m492.4/492.4 kB[0m [31m38.9 MB/s[0m eta [36m0:00:00[0m
    [?25h  Preparing metadata (setup.py) ... [?25l[?25hdone
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m268.8/268.8 kB[0m [31m25.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m7.8/7.8 MB[0m [31m86.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.3/1.3 MB[0m [31m66.5 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m66.0/66.0 kB[0m [31m7.4 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m375.1/375.1 kB[0m [31m36.1 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m57.3/57.3 kB[0m [31m6.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m115.3/115.3 kB[0m [31m13.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m212.5/212.5 kB[0m [31m22.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m134.8/134.8 kB[0m [31m15.6 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.6/1.6 MB[0m [31m78.9 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m103.2/103.2 kB[0m [31m11.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.1/1.1 MB[0m [31m62.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m81.0/81.0 kB[0m [31m9.7 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m62.6/62.6 kB[0m [31m7.8 MB/s[0m eta [36m0:00:00[0m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m1.2/1.2 MB[0m [31m60.6 MB/s[0m eta [36m0:00:00[0m
    [?25h[33mWARNING: jsonschema 4.3.3 does not provide the extra 'format-nongpl'[0m[33m
    [2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m66.4/66.4 kB[0m [31m8.1 MB/s[0m eta [36m0:00:00[0m
    [?25h  Building wheel for rouge_score (setup.py) ... [?25l[?25hdone
      Building wheel for pyrouge (setup.py) ... [?25l[?25hdone
      Building wheel for wget (setup.py) ... [?25l[?25hdone
    [31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
    google-colab 1.0.0 requires requests==2.27.1, but you have requests 2.31.0 which is incompatible.[0m[31m
    [0m

## Utils


```python
import requests
import json
import ipywidgets as widgets
from ipywidgets import Layout, Button, Box, FloatText, Textarea, Dropdown, Label, IntSlider, GridspecLayout
from IPython.display import display, Markdown, clear_output
import numpy as np
import pprint
import nltk
from scipy.stats import ttest_rel , ttest_ind, wilcoxon


def get_summ_example():
    global human_eval_data, current_doc_idx, num_of_eval_docs

    found = False
    for pos in range( current_doc_idx, len(human_eval_data) ):
        if human_eval_data[pos]["ranking_results"]["overall"] == [1,1]:
            human_eval_data[pos]["new_human_eval_results"] = human_eval_data[pos]["ranking_results"]
            num_of_eval_docs += 1
            if num_of_eval_docs >= len(human_eval_data):
                reveal_button.disabled = True
                submit_button.disabled = True

        else:
            found = True
            current_doc_idx = pos
            break
    if found:
        summ_example = human_eval_data[current_doc_idx]
        current_doc_idx = min(current_doc_idx+1, len(human_eval_data) )
    else:
        summ_example = None
        current_doc_idx = len(human_eval_data)

    return summ_example


class TextHTML(widgets.HTML):
    def __init__(self, html_style = {} ,**kwargs):
        super().__init__(**kwargs )
        self.default_html_style = {
            "padding":"5px",
            "height":"600px",
            "overflow-x":"hidden",
            "border":"1px solid grey",
            "line-height":"20px"
         }
        self.render_sen_list(html_style=html_style)
        self.html_lines = []

    def render_sen_list(self, sens=[], html_style = {}):
        self.default_html_style.update(html_style)
        html_lines = [
            '''<div style="%s">''' %( "; ".join( ":".join([key, value]) for key, value in self.default_html_style.items() )  )
        ]

        for sen in sens:
            is_marked = sen.get("is_marked", False)
            sen_text = sen.get("text", "").capitalize()
            html_line = '''<p> %s %s %s</p>'''%( '''<span style="background-color: #FFFF00">''' if is_marked else "",
                                             sen_text,
                                             '''</span>''' if is_marked else ""
                                           )
            html_lines.append( html_line )

        html_lines.append( "</div>" )
        value = "\n".join(html_lines)
        self.value = value
        self.html_lines = html_lines

    def update_html_style( self, html_style = {} ):
        self.default_html_style.update(html_style)
        self.html_lines[0] = '''<div style="%s">''' %( "; ".join( ":".join([key, value]) for key, value in self.default_html_style.items() )  )
        value = "\n".join(self.html_lines)
        self.value = value


form_item_layout = Layout(
    display='flex',
    flex_flow='row',
    justify_content='space-between'
)

rb_criteria = {}
for criterion in ["Overall:", "Coverage:", "Non-Redundancy:"]:
    rb_criteria[criterion] =  widgets.RadioButtons(
                options=['summary A', 'summary B'],
                disabled=False,
                index = None
    )

b_summ_sources = {}
colors_for_b_summ_sources ={ "Reference Summary":"YellowGreen","Summary A":"lightblue", "Summary B":"lightblue" }
for source in ["Reference Summary", "Summary A", "Summary B"]:
    b_summ_sources[source] = Button(description=source, layout=Layout(height='auto', width='auto'))
    b_summ_sources[source].style.button_color = colors_for_b_summ_sources[source]

text_summ_sources = {}
for source in ["Reference Summary", "Summary A", "Summary B"]:
    text_summ_sources[source] =  TextHTML({"height":"500px"})  # Textarea(layout=Layout(height="600px", width='auto'))

submit_button = Button(description="Submit & Eval Next", layout=Layout(height='auto', width='auto'))
submit_button.style.button_color = "LightSalmon"

reveal_button = Button(description="Reveal Answers\n(releval the choices of human annotators in original paper)", layout=Layout(height='auto', width='auto'))
reveal_button.style.button_color = "LightGreen"


report_button = Button(description="Report Human Evaluation Results", layout=Layout(height='auto', width='auto'))
report_button.style.button_color = "Salmon"



fulltext_button = Button(description="Show Source Document >>>", layout=Layout(height='auto', width='32.9%'))
fulltext_textbox = TextHTML( html_style={"height":"0px"}, layout=Layout(visibility="hidden") )

grid_b_summ = GridspecLayout(1,3)
grid_b_summ[0,0] = b_summ_sources["Reference Summary"]
grid_b_summ[0,1] = b_summ_sources["Summary A"]
grid_b_summ[0,2] = b_summ_sources["Summary B"]
grid_text_summ = GridspecLayout(1,3)
grid_text_summ[0,0] = text_summ_sources["Reference Summary"]
grid_text_summ[0,1] = text_summ_sources["Summary A"]
grid_text_summ[0,2] = text_summ_sources["Summary B"]
grid_rb_description = GridspecLayout(1,3)
grid_rb_description[0,0] = Label(value = "Overall:")
grid_rb_description[0,1] = Label(value = "Coverage (Information Integrity):")
grid_rb_description[0,2] = Label(value = "Non-Redundancy (Compactness):")

output_panel = widgets.Output()


form_items = [
    widgets.HTML(value = f"<b><font color='black' font size='4pt'>Read</b>"),


    grid_b_summ,
    grid_text_summ,
    fulltext_button,
    fulltext_textbox,
    widgets.HTML(value = f"<b><font color='black' font size='4pt'>Evaluation (choose one that is closer to the reference summary)</b>"),
    reveal_button,
    grid_rb_description,
    widgets.HBox([ rb_criteria["Overall:"], rb_criteria["Coverage:"], rb_criteria["Non-Redundancy:"]  ], layout=form_item_layout),
    widgets.Box([  submit_button, report_button ], layout=form_item_layout),
    output_panel

]

gui = Box(form_items, layout=Layout(
    display='flex',
    flex_flow='column',
    border='solid 2px',
    align_items='stretch',
    width='100%'
))


def get_next_example():
    global summ_example, num_of_eval_docs, human_eval_data
    summ_example = get_summ_example()
    if summ_example is not None:
        text_summ_sources["Reference Summary"].render_sen_list( [{"text":_} for _ in summ_example["summary"]] )
        text_summ_sources["Summary A"].render_sen_list( [{"text":_} for _ in summ_example["random_extracted_results"][0][0] ] )
        text_summ_sources["Summary B"].render_sen_list( [{"text":_} for _ in summ_example["random_extracted_results"][1][0] ] )
    else:
        text_summ_sources["Reference Summary"].render_sen_list( [] )
        text_summ_sources["Summary A"].render_sen_list( [] )
        text_summ_sources["Summary B"].render_sen_list( [] )


    for criterion in ["Overall:", "Coverage:", "Non-Redundancy:"]:
        rb_criteria[criterion].index = None

    if summ_example is not None:
        fulltext_textbox.render_sen_list( [{"text":_} for _ in summ_example["text"]] )
    else:
        fulltext_textbox.render_sen_list( [] )

    fulltext_textbox.update_html_style({"height":"0px"})
    fulltext_textbox.layout.visibility = "hidden"
    fulltext_button.description = "Show Source Document >>>"





def fulltext_button_on_click_listener(_):
    if fulltext_button.description == "Show Source Document >>>":
        fulltext_textbox.update_html_style({"height":"600px"})
        fulltext_textbox.layout.visibility = "visible"
        fulltext_button.description = "Hide Source Document >>>"
    elif fulltext_button.description == "Hide Source Document >>>":
        fulltext_textbox.update_html_style({"height":"0px"})
        fulltext_textbox.layout.visibility = "hidden"
        fulltext_button.description = "Show Source Document >>>"
fulltext_button.on_click( fulltext_button_on_click_listener )


def reveal_button_on_click_listener(_):
    global summ_example
    if summ_example is None:
        with output_panel:
            clear_output()
            print("No example is shown. Perhaps you have evaluated all data.")
    else:
        two_orders = [ [1,2],[2,1] ]
        rb_criteria["Overall:"].index = int(np.argmax([summ_example["ranking_results"]["overall"] == item for item in two_orders]))
        rb_criteria["Coverage:"].index = int(np.argmax([summ_example["ranking_results"]["coverage"] == item for item in two_orders]))
        rb_criteria["Non-Redundancy:"].index = int(np.argmax([summ_example["ranking_results"]["non-redundancy"] == item for item in two_orders]))

reveal_button.on_click( reveal_button_on_click_listener )

def submit_button_on_click_listener(_):
    global summ_example, num_of_eval_docs
    all_evaluated = True
    for criterion in ["Overall:", "Coverage:", "Non-Redundancy:"]:
        if rb_criteria[criterion].index is None:
            with output_panel:
                clear_output()
                print("You have not evaluated %s, please retry."%( criterion.rstrip(":") ))
            all_evaluated = False
    if all_evaluated:
        two_orders = [ [1,2],[2,1] ]
        summ_example["new_human_eval_results"] = {
                                         "overall":two_orders[ rb_criteria["Overall:"].index ] ,
                                         "coverage": two_orders[ rb_criteria["Coverage:"].index ],
                                         "non-redundancy":two_orders[ rb_criteria["Non-Redundancy:"].index ]
                                    }
        num_of_eval_docs += 1
        if num_of_eval_docs >= len(human_eval_data):
            reveal_button.disabled = True
            submit_button.disabled = True
        else:
            get_next_example()
        with output_panel:
            clear_output()
            print("You have evaluated %d/%d examples."%( num_of_eval_docs, len(human_eval_data)))

submit_button.on_click( submit_button_on_click_listener )


def report_results_on_click_listener(_):
    global human_eval_data
    word_tok = nltk.RegexpTokenizer(r'\w+')

    results = {}
    summ_len_key = "summary_length (# of sentences)"
    results[summ_len_key] = {}

    summ_words_len_key = "summary_length (# of words)"
    results[summ_words_len_key] = {}

    all_data = [ item for item in human_eval_data if "new_human_eval_results" in item ]

    for data in all_data:
        random_orders = data["random_orders"]
        ranking_results = data["ranking_results"]
        extracted_results = data["random_extracted_results"]
        for criterion in ranking_results:
            if criterion not in results:
                results[criterion] = {}
            for pos in range(len(random_orders)):
                results[criterion][random_orders[pos]] = results[criterion].get(random_orders[pos],[]) + [ ranking_results[criterion][pos] ]

        for pos in range(len(random_orders)):
            results[summ_len_key][random_orders[pos]] = results[summ_len_key].get(random_orders[pos],[]) + [ len(extracted_results[pos][1]) ]

        for pos in range(len(random_orders)):
            results[summ_words_len_key][random_orders[pos]] = results[summ_words_len_key].get(random_orders[pos],[]) + [ len( word_tok.tokenize(" ".join( [data["text"][_] for _ in extracted_results[pos][1] ])) ) ]
    with output_panel:
        clear_output()
        print("Human Evaluation Results:")
        print("Number of evaluated examples:",len(all_data))

        results_summ = {}
        for criterion in results:
            results_summ[criterion] ={}
            sample_list = []
            for model in results[criterion]:
                sample_list.append(results[criterion][model])
                results_summ[criterion][model] = np.mean(results[criterion][model])
            try:
                results_summ[criterion]["pvalue"] = wilcoxon(sample_list[0], sample_list[1]).pvalue
            except:
                print("Warning: too few samples to compute p value.")

        # print(results_summ)
        pprint.pprint(results_summ)



report_button.on_click(report_results_on_click_listener)


human_eval_data = None
summ_example = None
num_of_eval_docs = 0
current_doc_idx = 0

def run_gui( dataset_path,  width ="90%", textbox_height = "400px", ):
    global human_eval_data, summ_example, num_of_eval_docs, current_doc_idx
    human_eval_data = [ json.loads(line) for line in open(dataset_path,"r") ]

    summ_example = None
    num_of_eval_docs = 0
    current_doc_idx = 0
    get_next_example()
    with output_panel:
        clear_output()
        print("You have evaluated %d/%d examples."%( num_of_eval_docs, len(human_eval_data)))


    for source in ["Reference Summary", "Summary A", "Summary B"]:
        text_summ_sources[source].update_html_style({"height":textbox_height})
    gui.layout.width = width
    return gui

```

## Human Evaluation Experiment I:
MemSum v.s. NeuSum


```python
run_gui(dataset_path = "human_eval_results/records_memsum_neusum.jsonl")
```


    Box(children=(HTML(value="<b><font color='black' font size='4pt'>Read</b>"), GridspecLayout(children=(Button(d…


## Human Evaluation Experiment II:
MemSum w/o autostop v.s. NeuSum

**NOTE**: Experiment I and II cannot run at the same time on the same jupter notebook, because different ipywidgets share the same global variables.


```python
run_gui(dataset_path = "human_eval_results/records_memsum_wo_autostop_neusum.jsonl")
```


    Box(children=(HTML(value="<b><font color='black' font size='4pt'>Read</b>"), GridspecLayout(children=(Button(d…



```python

```
