{smcl}
{* *! version 17.0 27jun2024}{...}
{viewerdialog bbandit_initialize "dialog bbandit_initialize"}{...}
{viewerjumpto "Syntax" "bbandit_initialize##syntax"}{...}
{viewerjumpto "Description" "bbandit_initialize##description"}{...}
{viewerjumpto "Options" "bbandit_initialize##options"}{...}
{viewerjumpto "Examples" "bbandit_initialize##examples"}{...}

{p2col:{bf:bbandit_initialize}} Initialize a multi-armed bandit experiment

{marker syntax}{...}
{title:Syntax}
{p}
{cmd:bbandit_initialize} [, {it:batch(int 3)} {it:arms(int 2)} {it:exploration_phase(int 1)}]

{marker description}{...}
{title:Description}
{pstd}
{cmd:bbandit_initialize} sets up the initial data structure for running a multi-armed bandit experiment. 
This includes initializing the necessary variables and setting up the experimental conditions.

{pstd}
The program performs the following steps:

{phang2}
1. Captures the current dataset into a Python dictionary.

{phang2}
2. Converts the dictionary to a Pandas DataFrame.

{phang2}
3. Initializes the multi-armed bandit experiment with the specified parameters (batches, arms, exploration phase).

{phang2}
4. Creates the variables {bf:reward}, {bf:chosen_arm}, and {bf:batch} in Stata and initializes them to missing values.

{phang2}
5. Stores the results from the initialized DataFrame back into Stata variables.

{pstd}
The underlying algorithms are implemented in Python; therefore, a Python installation is necessary.

{marker options}{...}
{title:Options}
{phang}
{opt b:atch(int)} specifies the number of batches to divide the experiment into. The default is 3.

{phang}
{opt a:rms(int)} specifies the number of treatment arms in the bandit experiment. The default is 2.

{phang}
{opt e:xploration_phase(int)} specifies the number of batches where the algorithm assigns the treatment uniformly. The default value is 1.

{marker examples}{...}
{title:Examples}
{hline}
{pstd}
Setup the multi-armed bandit experiment with default values:

{phang2}{cmd:. bbandit_initialize}

{pstd}
Setup with specified number of batches, arms, and exploration phase:

{phang2}{cmd:. bbandit_initialize, Batches(5) Arms(3) Exploration_phase(2)}

{hline}
{pstd}
Authors: Jan Kemper, Davud Rostam-Afschar


