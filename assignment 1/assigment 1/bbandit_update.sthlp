{smcl}
{* *! version 17.0 27jun2024}{...}
{viewerdialog bbandit_update "dialog bbandit_update"}{...}
{viewerjumpto "Syntax" "bbandit_update##syntax"}{...}
{viewerjumpto "Description" "bbandit_update##description"}{...}
{viewerjumpto "Options" "bbandit_update##options"}{...}
{viewerjumpto "Examples" "bbandit_update##examples"}{...}

{p2col:{bf:bbandit_update}} Update a multi-armed bandit experiment

{marker syntax}{...}
{title:Syntax}
{p}
{cmd:bbandit_update} {it:varlist} [, {it:thompson} {it:greedy} {it:clipping(real 0.05)} {it:epsilon(real 0.1)} {it:seed(integer 1234)} {it:excel}]

{marker description}{...}
{title:Description}
{pstd}
{cmd:bbandit_update} updates the data structure for running a multi-armed bandit experiment based on the provided reward, chosen arm, and batch variables. This includes re-encoding chosen arm values, performing the specified updating algorithm (Thompson Sampling or epsilon-Greedy), and preparing the data for the next batch.

{pstd}
The program performs the following steps:

{phang2}
1. Re-encodes the chosen arm variable to be zero-indexed.

{phang2}
2. Converts the reward, chosen arm, and batch variables to numpy arrays.

{phang2}
3. Creates a Pandas DataFrame and corrects NaN values.

{phang2}
4. Preprocesses data for the specified updating algorithm (Thompson Sampling or epsilon-Greedy).

{phang2}
5. Executes the selected updating algorithm and updates the randomization or shuffling based on the results.

{phang2}
6. Initializes the updated reward, chosen arm, and batch variables in Stata.

{phang2}
7. Stores the updated variables back into the Stata dataset.

{phang2}
8. Optionally exports the dataset to an Excel file if the {opt Excel} option is specified.

{pstd}
The underlying algorithms are implemented in Python; therefore, a Python installation is necessary. The clipping rate or epsilon rate
has to be sufficiently high so that each arm is at least played once, otherwise there might be a python error.

{marker options}{...}
{title:Options}
{phang}
{opt t:hompson} specifies Bernoulli Thompson sampling algorithm.

{phang}
{opt g:reedy} specifies the epsilon-greedy algorithm.

{phang}
{opt c:lipping(real)} specifies the clipping rate for the Bernoulli Thompson algorithm. The default value is 0.05.

{phang}
{opt e:ps(real)} specifies the epsilon rate for the epsilon-greedy algorithm. The default value is 0.1.

{phang}
{opt ex:cel("path")} indicates that the updated data is saved as an Excel file under the specified path. The saved file can be used to impute the newly observed rewards.

{marker examples}{...}
{title:Examples}
{hline}
{pstd}
Update the multi-armed bandit experiment using the Thompson Sampling algorithm:

{phang2}{cmd:. bbandit_update reward chosen_arm batch, thompson}

{pstd}
Update the multi-armed bandit experiment using the epsilon-Greedy algorithm with specified epsilon and seed:

{phang2}{cmd:. bbandit_update reward chosen_arm batch, greedy epsilon(0.2)}

{pstd}
Update and export the dataset to an Excel file:

{phang2}{cmd:. bbandit_update reward chosen_arm batch, greedy excel("path")}

{hline}
{pstd}
Authors: Jan Kemper, Davud Rostam-Afschar
