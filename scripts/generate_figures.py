
import matplotlib
import json
from utils_functions import plottingClass
import os

### Figures style:
font = {'family': 'DejaVu Sans'}
matplotlib.rc('font', **font)
font_color = 'black'
matplotlib.rc('axes', edgecolor=font_color)
matplotlib.rcParams['xtick.color'] = font_color
matplotlib.rcParams['ytick.color'] = font_color
matplotlib.rcParams['text.color'] = font_color
matplotlib.rcParams['axes.labelcolor'] = font_color
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams.update({'figure.autolayout': True})

#### Define directories and load dynamics into results_dict :

figures_dir = 'figures/'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

json_file_name = 'brain_focus_dynamics.json'
with open(json_file_name) as json_file:
    results_dict = json.load(json_file)




#### Define which plots to generate:
plot_to_generate = {'plot_individual_dynamics': True, # Fig3.
                    'plot_model_performance': True, #Fig5.
                    'plot_avg_focus_statistics': True, #Fig6
                    'plot_max_session_distribution': True, #Fig6
                    'plot_time_dynamics_statistics': True} #Fig7




pltClass = plottingClass(figures_dir,results_dict)

##### Figure 3 - Plot individual dynamics:
if plot_to_generate['plot_individual_dynamics']:

    participants_ids = list(results_dict.keys())
    for participant_i,participant_id in enumerate(participants_ids):
        pltClass.plot_individual_dynamics(results_dict,participant_id,participant_i)


#### Figure 5 - Plot focus model performace
if plot_to_generate['plot_model_performance']:
    pltClass.plot_model_performance(results_dict)


##### Figure 6 - Average focus scores and statistical results for all and per subgroup

# Generates a matrix with the median scores per participant per session (51x4) - used to generate figure 6.
pltClass.generate_avg_results_matrix(results_dict)

# Define the subgroups to compare for Fig
groups_to_check = ['all', 'working', 'not-working', 'old', 'young']
groups_title = {'all': 'All','working': 'Working', 'not-working': 'Not Working', 'males': 'Males', 'females': 'Females',
                'old': 'Age > 36', 'young': 'Age < 36'}

## First row of Fig.6:
if plot_to_generate['plot_avg_focus_statistics']:
    pltClass.plot_avg_statistics(groups_to_check,groups_title)
## Second row of Fig6.
if plot_to_generate['plot_max_session_distribution']:
    pltClass.plot_max_session_distribution(groups_to_check,groups_title)


#### Figure 7 - Comparison of the focus dynamics between each pair + statistical analysis:
if plot_to_generate['plot_time_dynamics_statistics']:
    all_pairs2plot = [['endel', 'silence'], ['spotify', 'silence'], ['apple', 'silence'],
                      ['endel', 'spotify'], ['endel', 'apple'], ['apple', 'spotify']]
    group2check = 'all'
    pltClass.plot_time_dynamics_statistics(results_dict,group2check,all_pairs2plot)