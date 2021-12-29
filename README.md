## Measuring And Modeling The Effect of Audio On Human Focus In Everyday Environments Using Brain-Computer Interface Technology

This is a repository for the paper: Measuring And Modeling The Effect of Audio On Human Focus In Everyday Environments Using Brain-Computer Interface Technology. (doi: 10.3389/fncom.2021.760561)

The goal of this study was to learn what properties of sound affect human focus the most.

To do so, brain activity from 51 participants was analyzed. Participants recorded their brain activity using a wearable device (Muse-S by Interaxon) at home while performing various tasks and listening to different audio streams. 

During task performance, participants listened to one of four different background sounds (one per day, including Silence, Music playlists by Apple and Spotify, and personalized soundscapes by Endel). 

Brain decoding technology (neuOS by Arctop) was applied to transform the brain sensor data to predicted focus level dynamics at a rate of 5Hz over the course of each 30 minute session when participants performed a Preferred Task. 

The data and scripts in this repository demonstrate both how we validated the focus models predictions, and how we compared between the focus scores elicited by the different background sounds (including the statistical analysis).

Specifically, this repository contains:

* data/brain_focus_dynamics.json - The projected brain decoded focus dynamics per participant (N=51) for each audio stream while performing the Preferred Task (30 minutes). For each participant the json file also contains demographic information (age, gender), the Preferred Task type (working, reading, etc.), the self-reported focus rankings (survey results), and correlation scores of the selected model (between the predicted score and the rankings).
* scripts/generate_figures.py - main script to generate paper’s figures.
* scripts/utils_functions.py - contains a class with helper functions to generate figures and run statistical analysis.

For further details please contact: df@arctop.com
