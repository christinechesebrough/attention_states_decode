This folder contains data on (nearly) all of the patients and their implanted electrodes that the HBML laboratory has worked with.

Files
HBML_DATA.mat - File containing the data for the electrodes in matlab table format

ERRORS
List of subjects with errors (excluding those missing correspondence sheet or clinical info). Most of these subjects are not within the electrode table

* NS001 - Has no elec_recon folder

* NS003 - Has no elec_recon folder

* NS023 - Must double check the spec of each electrode

* NS029 - Has no elec_recon folder

* NS056 - Grid electrode must be configured differently due to how contacts were connected. There's a README file in subject's elec_recon folder to describe the issue

* NS069_postres - Has no elec_recon folder

* NS072 - Grid has some electrodes disabled

* NS073 - Error in running sub2AvgBrain due to some lesion. No average space coordinates could be generated so it is currently left out

* NS073_ORIG - Not added. But do we need?

* NS087_02 - Grid has some electrodes disabled

* NS088 - Error on doing PTD

* NS092 - Grid electrode was cut. Should possibly make a new mgrid with it split so that the correction looks better

* NS132_3 - Correspondence sheet and mgrid electrodes are different. Must resolve this large and concerning discrepency