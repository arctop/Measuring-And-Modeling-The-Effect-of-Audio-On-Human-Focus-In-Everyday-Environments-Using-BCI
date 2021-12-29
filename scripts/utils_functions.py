import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import pearsonr
import itertools
import pandas as pd
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_rel,ttest_ind
import matplotlib
import random
from sklearn.metrics import roc_auc_score, roc_curve,accuracy_score,confusion_matrix,balanced_accuracy_score


class plottingClass:

    def __init__(self,figures_dir,results_dict):
        ## Parameters:
        self.stride = 0.2 # in seconds (5Hz)
        self.time_hist = -10 # number of seconds before task beginning which are included in the dynamics
        self.task_length_min = 30 # task length in minutes
        self.colors_dict_audio = {'silence': 'gray', 'apple': 'm', 'endel': 'navy', 'spotify': 'green' } # color codes per audio stream
        self.figures_dir = figures_dir

        self.participants_ids = list(results_dict.keys())
        self.uni_audio_type = list(results_dict[self.participants_ids[0]]['Focus-brain decoded']) # unique audio types

    def plot_individual_dynamics(self,results_dict,participant_id,i):

        """
        Plots for a single participant the model dynamics per audio stream (4 sessions)
        ----- Parameters -----
        arg1 : results_dict - a dictionary containing all dynamics for all participants per audio stream and the preferred task type
        arg2 : participant_id - the participant_id (a string) - used to extract the specific participant data from the dict
        arg3 : i - the participant index (int) - used for the title only
        ------ Returns ------
        Saves the figure in the figures_dir
        """


        fig,ax = plt.subplots(len(self.uni_audio_type),1,figsize=(8,12))
        preferred_task_of_participant = results_dict[participant_id]['Preferred task type']
        for audio_i, audio_type in enumerate(self.uni_audio_type):
            curr_ev_pred = results_dict[participant_id]['Focus-brain decoded'][audio_type]
            curr_time_vec = np.linspace(self.time_hist, len(curr_ev_pred) * self.stride, len(curr_ev_pred))
            ax[audio_i].plot(curr_time_vec / 60, curr_ev_pred, color=self.colors_dict_audio[audio_type], linewidth=3)

            ax[audio_i].set_title(audio_type.capitalize() + ', Participant ' + str(
                i + 1) + '- Preferred Task (' + preferred_task_of_participant.capitalize() + ')',
                              fontsize=18, pad=10)
            ax[audio_i].set_xlabel('Time (min)')
            ax[audio_i].set_ylabel('Focus')
            ax[audio_i].set_ylim([0, 1])
            ax[audio_i].plot([self.time_hist / 60, self.task_length_min], [0.5, 0.5], '--k')


        plt.tight_layout()
        plt.savefig(self.figures_dir + 'preferred_task_dynamics_' + participant_id + '.png')
        plt.close()

        return None

    def plot_confusion_matrix(self,ax2plot,cm,classes,title,xlabel,ylabel):



        cmap = plt.cm.Blues
        ax2plot.imshow(cm, interpolation='nearest', cmap=cmap)
        ax2plot.set_title(title)
        ax2plot.set_xticks(np.arange(len(classes)))
        ax2plot.set_yticks(np.arange(len(classes)))
        ax2plot.set_xticklabels(classes, rotation=45)
        ax2plot.set_yticklabels(classes)

        fmt = '.2f'  # if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax2plot.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        ax2plot.set_xlabel(xlabel)
        ax2plot.set_ylabel(ylabel)



        return ax2plot

    def plot_model_performance(self,results_dict):

        """
        Plots a figure with 4 subplots demonstrating the model's performance
        A. Histograms of model performance per particpant (evaluated using the area under the ROC curve after thresholding the predictions)
        B. Scatter plot of the predicted focus score (brain decoded) vs. survey focus score - aggregated for all participants + audio types
        C. Confusion matrix for the global accuracy score after thresholding the predictions
        D. Average focus score per audio type (predicted vs. survey).
        ----- Parameters -----
        arg1 : results_dict - a dictionary containing all dynamics for all participants per audio stream, the survey focus scores, the models correlations and the correlations obtained by the shuffled survey scores
        ------ Returns ------
        Saves the figure in the figures_dir
        """

        all_focus_ranks = np.zeros([0])
        all_focus_preds = np.zeros([0])
        all_audio_type = np.zeros([0])
        all_participant_idx = np.zeros([0])
        auc_per_participant = np.zeros(len(self.participants_ids))
        for i,id in enumerate(self.participants_ids):
            for audio_i,audio_type in enumerate(self.uni_audio_type):
                curr_model_pred = np.array(results_dict[id]['Focus-brain decoded'][audio_type])
                curr_time_vec = np.linspace(self.time_hist, len(curr_model_pred) * self.stride, len(curr_model_pred))
                curr_rank = results_dict[id]['Focus-survey'][audio_type]
                all_focus_ranks = np.concatenate((all_focus_ranks,curr_rank),axis=0)
                # First half:
                pred_first_half = np.median(curr_model_pred[np.logical_and(curr_time_vec/60 < self.task_length_min/2,curr_time_vec > 0)])
                all_focus_preds = np.concatenate((all_focus_preds,[pred_first_half]),axis=0)
                # Second half
                pred_second_half = np.median(curr_model_pred[np.logical_and(curr_time_vec/60 > self.task_length_min/2,curr_time_vec < curr_time_vec[-1]-5)])
                all_focus_preds = np.concatenate((all_focus_preds,[pred_second_half]),axis=0)
                all_audio_type = np.concatenate((all_audio_type,2*[audio_type]),axis=0)
                all_participant_idx = np.concatenate((all_participant_idx,2*[i]),axis=0)

            curr_idx = all_participant_idx==i
            auc_per_participant[i] = roc_auc_score(y_true=all_focus_ranks[curr_idx]>np.mean(all_focus_ranks[curr_idx]),y_score=all_focus_preds[curr_idx])


        corr_aggregated = pearsonr(all_focus_preds,all_focus_ranks)
        print('Global pearson correlation:' + str(np.round(corr_aggregated,2)))
        print('<ROC>' + str(np.mean(auc_per_participant)))
        print('ste(ROC)' + str(np.std(auc_per_participant)/np.sqrt(len(auc_per_participant))))
        fig, ax = plt.subplots(2,2, figsize=(13, 11))

        ax[0,0].hist(auc_per_participant, alpha=0.5, density=False, color='b')
        ax[0,0].plot([0.5,0.5],[0,20], '--',alpha=0.5,color='k',linewidth=2)


        axins = inset_axes(ax[0,1], width="30%", height="30%", loc='upper left', bbox_to_anchor=(0.05, 0, 1, 1),
                           bbox_transform=ax[0,1].transAxes)

        th_vec = np.arange(0.2,0.8,0.066)
        plot_arrow_idx = np.arange(0,len(th_vec),3)
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        auc_score = np.zeros(len(th_vec))
        best_th = np.zeros(len(th_vec))
        cm_all_th = np.zeros([len(th_vec),2,2])
        acc_score = np.zeros(len(th_vec))
        for th_i in range(len(th_vec)):
            auc_score[th_i] = roc_auc_score(y_true=all_focus_ranks>th_vec[th_i],y_score=all_focus_preds)
            fpr, tpr, thresholds = roc_curve(all_focus_ranks>th_vec[th_i], all_focus_preds)
            best_th[th_i] = thresholds[np.argmin(fpr-tpr)]
            cm_all_th[th_i,:,:] = confusion_matrix(y_true=all_focus_ranks>th_vec[th_i],y_pred=all_focus_preds>best_th[th_i])
            acc_score[th_i] = balanced_accuracy_score(y_true=all_focus_ranks>th_vec[th_i],y_pred=all_focus_preds>best_th[th_i])
            if np.in1d(th_i,plot_arrow_idx):
                axins.plot(fpr,tpr,alpha=0.5,color=color_cycle[th_i])
                # ax[0,1].plot([th_vec[th_i],th_vec[th_i]],[0.2,0.8],'--',alpha=0.5)
                ax[0,1].arrow(th_vec[th_i],0.3,0,-0.1,width=0.01,alpha=0.25,color=color_cycle[th_i])
                # ax[0,1].plot([th_vec[th_i],th_vec[th_i]],[0.2,0.8],'--',alpha=0.5)
        axins.plot([0,1],[0,1],'--k')
        axins.set_ylabel('TP',labelpad=-15)
        axins.set_xlabel('FP',labelpad=-15)

        th_idx2plot = 2
        cm = cm_all_th[th_idx2plot,:,:].astype('float') / cm_all_th[th_idx2plot,:,:].sum(axis=1)[:, np.newaxis]

        classes = ['Low Focus', 'High Focus']
        title = 'Acc=' + str(np.round(acc_score[th_idx2plot],2)) +', Auc=' + str(np.round(auc_score[th_idx2plot],2))
        self.plot_confusion_matrix(ax[1,0],cm,classes,title,xlabel='Brain Decoded',ylabel='Self-Reported')


        mean_rank_per_audio = np.zeros(len(self.uni_audio_type))
        mean_score_per_audio = np.zeros(len(self.uni_audio_type))
        std_rank_per_audio = np.zeros(len(self.uni_audio_type))
        std_score_per_audio = np.zeros(len(self.uni_audio_type))
        for audio_i, audio_type in enumerate(self.uni_audio_type):

            curr_audio_idx = all_audio_type == audio_type
            mean_rank_per_audio[audio_i] = np.mean(all_focus_ranks[curr_audio_idx])
            mean_score_per_audio[audio_i] = np.mean(all_focus_preds[curr_audio_idx])
            std_rank_per_audio[audio_i] = np.std(all_focus_ranks[curr_audio_idx])/np.sqrt(np.sum(curr_audio_idx))
            std_score_per_audio[audio_i] = np.std(all_focus_preds[curr_audio_idx])/np.sqrt(np.sum(curr_audio_idx))


            ax[0,1].plot(all_focus_ranks[curr_audio_idx], all_focus_preds[curr_audio_idx], '.',
                       color=self.colors_dict_audio[audio_type])
            ax[1,1].plot(mean_rank_per_audio[audio_i], mean_score_per_audio[audio_i],
                       color=self.colors_dict_audio[audio_type],
                       marker='s', label=audio_type.capitalize(), markersize=14)
            ax[1,1].errorbar(mean_rank_per_audio[audio_i], mean_score_per_audio[audio_i],
                           xerr=std_rank_per_audio[audio_i],
                           yerr=std_score_per_audio[audio_i], color=self.colors_dict_audio[audio_type])

        a, b = np.polyfit(mean_rank_per_audio.flatten(), mean_score_per_audio.flatten(), 1)
        x = np.array([np.min(mean_rank_per_audio) - np.max(std_rank_per_audio),
                      np.max(mean_rank_per_audio) + np.max(std_rank_per_audio)])
        x_trans = a * (x) + b
        ax[1,1].plot(x, x_trans, '--k', linewidth=3)
        ax[0,0].set_xlabel('Models AUC-ROC')
        ax[0,0].set_ylabel('# Participants')
        ax[0,0].set_title('AUC-ROC per participant', fontsize=20)

        ax[0,1].set_xlabel('Self Reported')
        ax[0,1].set_ylabel('Brain Decoded')
        ax[0,1].set_title('Focus Scores - Preferred Task', fontsize=20)

        ax[1,1].set_xlabel('Self Reported - Average')
        ax[1,1].set_ylabel('Brain Decoded - Average')
        ax[1,1].set_title('Average Focus Score Per Audio', fontsize=20)
        ax[1,1].legend(ncol=2, fontsize=12)
        plt.tight_layout()

        fig.align_xlabels(ax[1,:])
        plt.savefig(self.figures_dir + 'models_proof_subplot_accuracy.png')
        plt.close()

        return None

    def plot_model_performance_corr(self,results_dict):

        """
        Plots a figure with 3 subplots demonstrating the model's performance
        A. Histograms of model performance per particpant (evaluated using a correlation score), compared to performance based on shuffled survey scores
        B. Scatter plot of the predicted focus score (brain decoded) vs. survey focus score - aggregated for all participants + audio types
        C. Average focus score per audio type (predicted vs. survey).
        ----- Parameters -----
        arg1 : results_dict - a dictionary containing all dynamics for all participants per audio stream, the survey focus scores, the models correlations and the correlations obtained by the shuffled survey scores
        ------ Returns ------
        Saves the figure in the figures_dir
        """


        all_focus_ranks = np.zeros([0])
        all_focus_preds = np.zeros([0])
        all_audio_type = np.zeros([0])
        for i,id in enumerate(self.participants_ids):
            for audio_i,audio_type in enumerate(self.uni_audio_type):
                curr_model_pred = np.array(results_dict[id]['Focus-brain decoded'][audio_type])
                curr_time_vec = np.linspace(self.time_hist, len(curr_model_pred) * self.stride, len(curr_model_pred))
                curr_rank = results_dict[id]['Focus-survey'][audio_type]
                all_focus_ranks = np.concatenate((all_focus_ranks,curr_rank),axis=0)
                # First half:
                pred_first_half = np.median(curr_model_pred[np.logical_and(curr_time_vec/60 < self.task_length_min/2,curr_time_vec > 0)])
                all_focus_preds = np.concatenate((all_focus_preds,[pred_first_half]),axis=0)
                pred_second_half = np.median(curr_model_pred[np.logical_and(curr_time_vec/60 > self.task_length_min/2,curr_time_vec < curr_time_vec[-1]-5)])
                all_focus_preds = np.concatenate((all_focus_preds,[pred_second_half]),axis=0)
                all_audio_type = np.concatenate((all_audio_type,2*[audio_type]),axis=0)


        corr_aggregated = pearsonr(all_focus_preds,all_focus_ranks)

        corr_best_shuffle = np.array([results_dict[x]['Model-shuffle-corr'] for x in self.participants_ids])
        corr_best = np.array([results_dict[x]['Model-corr'] for x in self.participants_ids])

        fig, ax = plt.subplots(1, 3, figsize=(16, 5))
        ax[0].hist(corr_best, alpha=0.5, label='Real', density=True, color='b')
        ax[0].hist(corr_best_shuffle.flatten(), alpha=0.5, label='Shuffle', density=True, color='gray')
        avg_corr_best = np.nanmean(corr_best)
        avg_corr_shuffle = np.nanmean(corr_best_shuffle)
        std_corr_best = np.nanstd(corr_best) / np.sqrt(len(corr_best))
        std_corr_shuffle = np.nanstd(corr_best_shuffle) / np.sqrt(len(corr_best))
        axins = inset_axes(ax[0], width="30%", height="30%", loc='upper left', bbox_to_anchor=(0.1, 0, 1, 1),
                           bbox_transform=ax[0].transAxes)
        axins.bar(1, avg_corr_shuffle, yerr=std_corr_shuffle, alpha=0.5, color='gray')
        axins.bar(2, avg_corr_best, yerr=std_corr_best, alpha=0.5, color='b')

        mean_rank_per_audio = np.zeros(len(self.uni_audio_type))
        mean_score_per_audio = np.zeros(len(self.uni_audio_type))
        std_rank_per_audio = np.zeros(len(self.uni_audio_type))
        std_score_per_audio = np.zeros(len(self.uni_audio_type))
        for audio_i, audio_type in enumerate(self.uni_audio_type):

            curr_audio_idx = all_audio_type == audio_type
            mean_rank_per_audio[audio_i] = np.mean(all_focus_ranks[curr_audio_idx])
            mean_score_per_audio[audio_i] = np.mean(all_focus_preds[curr_audio_idx])
            std_rank_per_audio[audio_i] = np.std(all_focus_ranks[curr_audio_idx])/np.sqrt(np.sum(curr_audio_idx))
            std_score_per_audio[audio_i] = np.std(all_focus_preds[curr_audio_idx])/np.sqrt(np.sum(curr_audio_idx))


            ax[1].plot(all_focus_ranks[curr_audio_idx], all_focus_preds[curr_audio_idx], '.',
                       color=self.colors_dict_audio[audio_type])
            ax[2].plot(mean_rank_per_audio[audio_i], mean_score_per_audio[audio_i],
                       color=self.colors_dict_audio[audio_type],
                       marker='s', label=audio_type.capitalize(), markersize=14)
            ax[2].errorbar(mean_rank_per_audio[audio_i], mean_score_per_audio[audio_i],
                           xerr=std_rank_per_audio[audio_i],
                           yerr=std_score_per_audio[audio_i], color=self.colors_dict_audio[audio_type])

        a, b = np.polyfit(mean_rank_per_audio.flatten(), mean_score_per_audio.flatten(), 1)
        x = np.array([np.min(mean_rank_per_audio) - np.max(std_rank_per_audio),
                      np.max(mean_rank_per_audio) + np.max(std_rank_per_audio)])
        x_trans = a * (x) + b
        ax[2].plot(x, x_trans, '--k', linewidth=3)
        ax[0].legend(loc='upper right', fontsize=12)
        axins.set_xticks([])
        ax[0].set_xlabel('Models correlations')
        ax[0].set_ylabel('P(Models correlations)')
        ax[0].set_title('Correlation per participant', fontsize=20)
        ax[0].set_yticks([])
        ax[1].set_xlabel('Self-Reported')
        ax[1].set_ylabel('Brain Decoded')
        ax[1].set_title('Focus Scores - Preferred Task', fontsize=20)

        ax[2].set_xlabel('Self-Reported - Average')
        ax[2].set_ylabel('Brain Decoded - Average')
        ax[2].set_title('Average Focus Score Per Audio', fontsize=20)
        ax[2].legend(ncol=2, fontsize=12)

        plt.tight_layout()
        plt.savefig(self.figures_dir + 'models_validity_analysis.png')
        plt.close()

        return None

    def run_shuffle_cluster_analysis(self,dynamics_cond1, dynamics_cond2, statistics_time_points, p_val_th=0.05):

        """
        Runs a shuffle analysis for the statistical tests between two time dynamics to obtain a threshold for a minimum consecutive significant time points
        ----- Parameters -----
        arg1 : dynamics_cond1 - array (N participants x M time points - floats) - focus score per time sample for each participant in condition 1 (audio type 1)
        arg2 : dynamics_cond2 - array (N participants x M time points - floats) - focus score per time sample for each participant in condition 2 (audio type 2)
        arg3 : statistics_time_points - array (K time points - int) - indices of time points to sample and perform statistical analysis on
        ------ Returns ------
        biggest_cluster_th - the threshold for the minimum number of consecutive significant time points to consider the cluster as significant
        """

        n_subs1 = np.shape(dynamics_cond1)[0]
        n_subs2 = np.shape(dynamics_cond2)[0]
        cond_subs_vec = np.concatenate((0 * np.ones(n_subs1), 1 * np.ones(n_subs2)))
        dynamics_concat = np.concatenate((dynamics_cond1, dynamics_cond2), axis=0)

        n_iters = 1000
        biggest_cluster = np.nan * np.ones(n_iters)
        print_output =False
        for iter_i in range(n_iters):
            random.shuffle(cond_subs_vec)
            dynamics_cond1_shuffle = dynamics_concat[cond_subs_vec == 0]
            dynamics_cond2_shuffle = dynamics_concat[cond_subs_vec == 1]

            not_nan_idx1 = np.all(~np.isnan(dynamics_cond1_shuffle[:, statistics_time_points]), axis=1)
            not_nan_idx2 = np.all(~np.isnan(dynamics_cond2_shuffle[:, statistics_time_points]), axis=1)
            _, p_vals_shuffle = ttest_ind(dynamics_cond1_shuffle[not_nan_idx1, :][:, statistics_time_points],
                                          dynamics_cond2_shuffle[not_nan_idx2, :][:, statistics_time_points])

            significance_idx_shuffle = p_vals_shuffle < p_val_th

            significance_idx_shuffle = np.append(significance_idx_shuffle, False)
            significance_idx_shuffle[0] = False
            absdiff = np.abs(np.diff(significance_idx_shuffle))
            if np.sum(absdiff) == 0:
                biggest_cluster[iter_i] = 0
            try:
                ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
                biggest_cluster[iter_i] = np.max(ranges[:, 1] - ranges[:, 0])

            except:
                if print_output:
                    print('No significant clusters')

        biggest_cluster_th = np.percentile(biggest_cluster, 95)

        return biggest_cluster_th

    def generate_avg_results_matrix(self,results_dict):

        """
        Generates a summary matrix of a focus score per audio type for each pariticpant based on the median score during the task
        ----- Parameters -----
        arg1 : results_dict - a dictionary containing all dynamics for all participants per audio stream ,demograhpic info (age) and the preferred task type
        ------ Returns ------
        updates the class parameters:
        stat_per_participant - array (N participants x M audio types - float) - median focus score for each participant during each audio type
        age_vec - array (N participants - int) - the age for each participant
        preferred_task_vec - array (N participants - strings) - the type of the Preferred task for each participant (working, reading, playing, etc)

        """

        preferred_task_vec = np.zeros([0])
        stat_per_participant = np.nan * np.ones([len(self.participants_ids), len(self.uni_audio_type)])
        age_vec = np.zeros(len(self.participants_ids))

        for i,id in enumerate(self.participants_ids):
            age_vec[i] = results_dict[id]['Age']
            preferred_task_vec = np.concatenate((preferred_task_vec, [results_dict[id]['Preferred task type']]), axis=0)

            for audio_i,audio_type in enumerate(self.uni_audio_type):
                curr_model_pred = np.array(results_dict[id]['Focus-brain decoded'][audio_type])
                curr_time_vec = np.linspace(self.time_hist, len(curr_model_pred) * self.stride, len(curr_model_pred))
                pred_first_half = np.median(curr_model_pred[np.logical_and(curr_time_vec/60 < self.task_length_min/2,curr_time_vec > 0)])
                pred_second_half = np.median(curr_model_pred[np.logical_and(curr_time_vec/60 > self.task_length_min/2,curr_time_vec < curr_time_vec[-1]-5)])
                stat_per_participant[i,audio_i] = np.mean(np.array([pred_first_half,pred_second_half]))

        self.stat_per_participant = stat_per_participant
        self.age_vec = age_vec
        self.preferred_task_vec = preferred_task_vec

    def plot_avg_statistics(self,groups_to_check,groups_title):

        """
        Plots average focus score per audio type for each subgroup together with statistical results
        Statistical anaylsis included One-way Anove together with post-hoc tests with Holm-Bonferroni correction
        The data used to plot this figure is based on the class parameter self.stat_per_participant
        ----- Parameters -----
        arg1 : groups_to_check - a list with subgroups names to plot
        arg2 : groups_title - a dictionary which maps a group name with a title to show in the figure
        ------ Returns ------
        Saves the figure in the figures_dir
        """

        median_age = np.median(self.age_vec)

        all_pairs_table = list(itertools.combinations(np.arange(len(self.uni_audio_type)), 2))
        all_pairs_table = np.array([np.array(x) for x in all_pairs_table])
        pairs_to_test = [[self.uni_audio_type[x[0]], self.uni_audio_type[x[1]]] for x in all_pairs_table]

        all_p_vals_holm = {}
        fig, ax = plt.subplots(1,len(groups_to_check) ,figsize=(20, 5))
        anove_table = {}
        pairs_averages_table = {}
        for group_i, curr_group in enumerate(groups_to_check):

            if curr_group=='all':
                font_weight = 'bold'
            else:
                font_weight = 'normal'

            all_p_vals_holm[curr_group] = np.zeros([len(pairs_to_test)])

            rel_part_idx = np.all(~np.isnan(self.stat_per_participant), axis=1)
            if curr_group == 'working' or curr_group == 'reading':
                rel_part_idx = np.logical_and(rel_part_idx, self.preferred_task_vec == curr_group)
            elif curr_group == 'not-working':
                rel_part_idx = np.logical_and(rel_part_idx, self.preferred_task_vec != 'working')
            elif curr_group == 'old':
                rel_part_idx = np.logical_and(rel_part_idx, self.age_vec >= median_age)
            elif curr_group == 'young':
                rel_part_idx = np.logical_and(rel_part_idx, self.age_vec < median_age)

            stat_per_participant_rel = self.stat_per_participant[rel_part_idx]
            user_vec = np.repeat(np.where(rel_part_idx)[0], len(self.uni_audio_type))
            audio_vec_for_stat = np.tile(np.arange(len(self.uni_audio_type)), np.sum(rel_part_idx))
            emotion_vec_for_stat = stat_per_participant_rel.flatten()

            data_anova = {'user': user_vec,
                          'audio': audio_vec_for_stat,
                          'focus': emotion_vec_for_stat}

            df_anova = pd.DataFrame(data_anova)
            anove_fitted = AnovaRM(data=df_anova, depvar='focus', subject='user', within=['audio']).fit()
            anove_table[curr_group] =anove_fitted.anova_table
            p_anova =anove_fitted.anova_table['Pr > F'].values[0]
            print(anove_fitted)

            p_vals_before_correction = np.zeros(len(all_pairs_table))
            t_vals = np.zeros(len(all_pairs_table))
            df_vals = np.zeros(len(all_pairs_table))
            avg_vals = np.zeros([len(all_pairs_table), 2])
            std_vals = np.zeros([len(all_pairs_table), 2])

            avg_diff = np.zeros([len(all_pairs_table)])
            std_diff = np.zeros([len(all_pairs_table)])
            for pair_i, curr_pair in enumerate(all_pairs_table):
                t_vals[pair_i], p_vals_before_correction[pair_i] = ttest_rel(stat_per_participant_rel[:, curr_pair[0]],
                                                                             stat_per_participant_rel[:, curr_pair[1]])
                df_vals[pair_i] = np.shape(stat_per_participant_rel[:, curr_pair[0]])[0] - 1
                avg_vals[pair_i, 0] = np.mean(stat_per_participant_rel[:, curr_pair[0]])
                avg_vals[pair_i, 1] = np.mean(stat_per_participant_rel[:, curr_pair[1]])
                std_vals[pair_i, 0] = np.std(stat_per_participant_rel[:, curr_pair[0]])
                std_vals[pair_i, 1] = np.std(stat_per_participant_rel[:, curr_pair[1]])
                avg_diff[pair_i] = np.mean(
                    stat_per_participant_rel[:, curr_pair[0]] - stat_per_participant_rel[:, curr_pair[1]])
                std_diff[pair_i] = np.std(
                    stat_per_participant_rel[:, curr_pair[0]] - stat_per_participant_rel[:, curr_pair[1]])/np.sqrt(df_vals[pair_i])
            reject, p_corrected, alpha_sid, alpha_bonf = multipletests(p_vals_before_correction, alpha=0.05,
                                                                       method='holm')
            pairs_averages_table[curr_group] = {}
            pairs_averages_table[curr_group]['avg diff'] = avg_diff
            pairs_averages_table[curr_group]['std diff'] = std_diff
            pairs_averages_table[curr_group]['t val'] = t_vals
            pairs_averages_table[curr_group]['df val'] = df_vals
            pairs_averages_table[curr_group]['p'] = p_corrected
            pairs_averages_table[curr_group] = pd.DataFrame(pairs_averages_table[curr_group])

            mean_audio_rank = np.zeros(len(self.uni_audio_type))
            std_audio_rank = np.zeros(len(self.uni_audio_type))
            for audio_i, audio_type in enumerate(self.uni_audio_type):
                mean_audio_rank[audio_i] = np.mean(stat_per_participant_rel[:, audio_i])
                std_audio_rank[audio_i] = np.std(stat_per_participant_rel[:, audio_i]) / np.sqrt(
                    len(stat_per_participant_rel[:, audio_i]))
                ax[group_i].bar(audio_i, mean_audio_rank[audio_i], yerr=std_audio_rank[audio_i],
                                             color=self.colors_dict_audio[audio_type])


            h = 0.025
            y_diff = 0.03
            for pair_i, curr_pair in enumerate(all_pairs_table):

                curr_p_holm = p_corrected[pair_i]
                if curr_p_holm < 0.05 and p_anova < 0.05:
                    x1, x2 = curr_pair[0], curr_pair[
                        1]
                    y = np.max(mean_audio_rank) + y_diff
                    ax[group_i].plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, c='k')
                    y_diff = y_diff + 0.04
                    curr_str = 'p=' + str(np.round(curr_p_holm, 3))
                    ax[group_i].text((x1 + x2) * .5, y + h, curr_str, ha='center', va='bottom',
                                                  color='k', fontsize=12,fontweight=font_weight)



            ax[group_i].set_xticks(np.arange(len(self.uni_audio_type)))

            ax[group_i].set_title(groups_title[curr_group],fontdict={"fontweight":font_weight})
            ax[group_i].set_ylabel('Focus',fontdict={"fontweight":font_weight})
            ax[group_i].set_xticklabels(self.uni_audio_type, rotation=45,fontdict={"fontweight":font_weight})
            ax[group_i].set_yticks(ax[group_i].get_yticks())
            ax[group_i].set_yticklabels(np.round(ax[group_i].get_yticks(),1),fontdict={"fontweight":font_weight})

            ax[group_i].set_ylim([0.3,0.7])

        plt.tight_layout()
        plt.savefig(self.figures_dir + 'focus_average_statistics_analysis.png')
        plt.close()

    def plot_max_session_distribution(self,groups_to_check,groups_title):

        """
        Plots the distribution for the maximum focus score session
        ----- Parameters -----
        arg1 : groups_to_check - a list with subgroups names to plot
        arg2 : groups_title - a dictionary which maps a group name with a title to show in the figure
        ------ Returns ------
        Saves the figure in the figures_dir
        """

        pie_colors = [matplotlib.colors.to_rgb(self.colors_dict_audio[x]) for x in self.uni_audio_type]
        uni_audio_type_cap = [x.capitalize() for x in self.uni_audio_type]
        median_age = np.median(self.age_vec)

        fig, ax = plt.subplots(1,len(groups_to_check), figsize=(20, 6))
        for group_i, curr_group in enumerate(groups_to_check):
            if curr_group=='all':
                font_weight = 'bold'
            else:
                font_weight = 'normal'
            rel_part_idx = np.all(~np.isnan(self.stat_per_participant), axis=1)
            if curr_group == 'working' or curr_group == 'reading':
                rel_part_idx = np.logical_and(rel_part_idx, self.preferred_task_vec == curr_group)
            elif curr_group == 'not-working':
                rel_part_idx = np.logical_and(rel_part_idx, self.preferred_task_vec != 'working')
            elif curr_group == 'old':
                rel_part_idx = np.logical_and(rel_part_idx, self.age_vec >= median_age)
            elif curr_group == 'young':
                rel_part_idx = np.logical_and(rel_part_idx, self.age_vec < median_age)

            stat_per_participant_rel = self.stat_per_participant[rel_part_idx]
            sort_idx = np.zeros_like(stat_per_participant_rel)

            for participant_i in range(np.shape(stat_per_participant_rel)[0]):
                sort_idx[participant_i, :] = np.argsort(stat_per_participant_rel[participant_i]) + 1

            counts_per_audio = np.zeros([len(self.uni_audio_type), len(self.uni_audio_type)])
            for pos_i in range(len(self.uni_audio_type)):
                counts_per_audio[pos_i, :] = np.histogram(sort_idx[:, pos_i], bins=np.arange(0.5, 5.5))[0]
            wdge, texts, auttext = ax[group_i].pie(counts_per_audio[3, :],
                                                                labels=uni_audio_type_cap, colors=pie_colors,
                                                                startangle=45,
                                                                autopct='%1.0f%%', shadow=True,
                                                                textprops=dict(color="white", fontsize=16,fontweight=font_weight))
            for text in texts:
                text.set_color('k')
            ax[group_i].set_title(groups_title[curr_group], fontsize=20, pad=15,fontdict={"fontweight": font_weight})
        plt.tight_layout()
        plt.savefig(self.figures_dir + 'best_session_distribution_analysis.png')
        plt.close()

    def plot_time_dynamics_statistics(self,results_dict,group2check,all_pairs2plot):

        """
        Plots the focus dynamics during the 30 minutes of preffered task for different pairs of audio types.
        Includes statistical analysis between pairs to indicate areas with significant difference.
        ----- Parameters -----
        arg1 : group2check - a string with sugroup name to plot (in the paper - 'all' is used)
        arg2 : all_pairs2plot - list of lists containing pairs of audio types to compare between
        ------ Returns ------
        Saves the figure in the figures_dir
        """


        n_statistics = 30 * 60 # statistics second by second
        p_th = 0.05

        # Get minimum length to align dynamics:
        session_len = np.zeros([0])
        for i,id in enumerate(self.participants_ids):
            curr_len = [len(results_dict[id]['Focus-brain decoded'][x]) for x in self.uni_audio_type]
            session_len = np.concatenate((session_len, np.array(curr_len)))
        min_len = int(np.min(session_len))
        statistics_time_points = np.linspace(0, min_len - 1, n_statistics).astype(int)
        median_age = np.median(self.age_vec)
        fig, ax = plt.subplots(3, 2, figsize=(12, 12))
        biggest_cluster_th_vec = np.zeros(len(all_pairs2plot))
        significant_percent = np.zeros(len(all_pairs2plot))
        significant_start_min = np.zeros(len(all_pairs2plot))
        significant_start_max = np.zeros(len(all_pairs2plot))
        # Go over the pairs:
        for pair_i, pairs2plot in enumerate(all_pairs2plot):
            print(pair_i)
            ax_idx = np.unravel_index(pair_i, (3, 2))
            dynamics_aligns = {}
            for audio_i, audio_type in enumerate(pairs2plot):
                # align dynamics:
                curr_dynamics = [results_dict[x]['Focus-brain decoded'][audio_type] for x in self.participants_ids]
                curr_dynamics_aligned = []
                for i in range(len(curr_dynamics)):
                    curr_dynamics_aligned.append(curr_dynamics[i][:min_len])
                curr_dynamics_aligned = np.array(curr_dynamics_aligned)
                # get relevant user indices
                if group2check == 'all':
                    curr_rel_idx = np.arange(np.shape(curr_dynamics_aligned)[0])
                elif group2check == 'working':
                    curr_rel_idx = np.where(self.preferred_task_vec == 'working')[0]
                elif group2check == 'not-working':
                    curr_rel_idx = np.where(self.preferred_task_vec != 'working')[0]
                elif group2check == 'old':
                    curr_rel_idx = np.where(self.age_vec >= median_age)[0]
                elif group2check == 'young':
                    curr_rel_idx = np.where(self.age_vec < median_age)[0]

                dynamics_aligns[audio_type] = curr_dynamics_aligned[curr_rel_idx]
                avg_dynamics = np.nanmean(curr_dynamics_aligned[curr_rel_idx], axis=0)
                ste_dynamics = np.nanstd(curr_dynamics_aligned[curr_rel_idx], axis=0) / np.sqrt(len(curr_rel_idx))
                time_vec = np.linspace(self.time_hist, len(avg_dynamics) * self.stride, len(avg_dynamics)) / 60
                ax[ax_idx[0], ax_idx[1]].plot(time_vec, avg_dynamics, color=self.colors_dict_audio[audio_type],
                                              label=audio_type.capitalize())
                ax[ax_idx[0], ax_idx[1]].plot([time_vec[0], time_vec[-1]], [0.5, 0.5], '--k')
                ax[ax_idx[0], ax_idx[1]].fill_between(time_vec, avg_dynamics - ste_dynamics,
                                                      avg_dynamics + ste_dynamics,
                                                      facecolor=self.colors_dict_audio[audio_type], alpha=0.5)
            ax[ax_idx[0], ax_idx[1]].set_ylim([0.3, 0.7])
            ax[ax_idx[0], ax_idx[1]].set_title(pairs2plot[0].capitalize() + ' vs. ' + pairs2plot[1].capitalize())
            ax[ax_idx[0], ax_idx[1]].set_ylabel('Focus')
            ax[ax_idx[0], ax_idx[1]].set_xlabel('Time (min)')
            ax[ax_idx[0], ax_idx[1]].legend()

            not_nan_idx1 = np.all(~np.isnan(dynamics_aligns[pairs2plot[0]][:, statistics_time_points]), axis=1)
            not_nan_idx2 = np.all(~np.isnan(dynamics_aligns[pairs2plot[1]][:, statistics_time_points]), axis=1)
            _, p_vals_before_correction = ttest_ind(
                dynamics_aligns[pairs2plot[0]][not_nan_idx1, :][:, statistics_time_points],
                dynamics_aligns[pairs2plot[1]][not_nan_idx2, :][:, statistics_time_points])

            biggest_cluster_th = self.run_shuffle_cluster_analysis(dynamics_aligns[pairs2plot[0]],
                                                              dynamics_aligns[pairs2plot[1]],
                                                              statistics_time_points, p_th)
            biggest_cluster_th_vec[pair_i] = biggest_cluster_th

            significance_idx_before = p_vals_before_correction < p_th

            significance_idx_before = np.append(significance_idx_before, False)
            significance_idx_before[0] = False
            absdiff = np.abs(np.diff(significance_idx_before))
            try:
                ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
            except:
                print('No clusters')
            len_per_cluster = ranges[:, 1] - ranges[:, 0]
            good_clusters = ranges[len_per_cluster > biggest_cluster_th]
            time_sign = 0
            for cluster_i in range(len(good_clusters)):
                curr_time_clust = time_vec[statistics_time_points[good_clusters[cluster_i]]]
                ax[ax_idx[0], ax_idx[1]].fill_between(curr_time_clust, [-1, -1], [1, 1], facecolor='black', alpha=0.3)
                if cluster_i == 0:
                    significant_start_min[pair_i] = curr_time_clust[0]
                    significant_start_max[pair_i] = curr_time_clust[1]

                time_sign = time_sign + (curr_time_clust[1] - curr_time_clust[0])
            significant_percent[pair_i] = (time_sign / 30)

        plt.tight_layout()
        plt.savefig(self.figures_dir + 'time_dynamics_analysis.png')
        plt.show()
        plt.close()