{smcl}
{* *! version 17.0 30jun2024}{...}
{viewerdialog bbandits "dialog bbandits"}{...}
{viewerjumpto "Syntax" "bbandits##syntax"}{...}
{viewerjumpto "Menu" "bbandits##menu"}{...}
{viewerjumpto "Description" "bbandits##description"}{...}
{viewerjumpto "Options" "bbandits##options"}{...}
{viewerjumpto "Examples" "bbandits##examples"}{...}

{p2col:{bf:bbandits}} Bandit Inference with Thompson Sampling and other methods

{marker syntax}{...}
{title:Syntax}
{p}
{cmd:bbandits} reward assignedarm batch [, {opt reference_arm(int 0)} {opt test_value(real 0.0)} {opt plot_thompson}  opt{stacked}  opt{twooptions_Thompson(string)}
opt{twooptions_bols(string)}  opt{twooptions_ols(string)}
opt{twooptions_sharebybatch(string)}  opt{twoptions_stackedsharebybatch(string)}
opt{twooptions_cumsharesbyybatch(string)}]

{marker description}{...}
{title:Description}
{pstd}
{cmd:bbandits} performs bandit inference using the specified data and options. The data is processed in Stata and then analyzed using Python functions. The results are returned and stored in Stata matrices for further analysis. 
To calculate the BOLS estimates and confidence intervals, each arm should be played at least once in each batch. 
Otherwise, python will return an error like "ZeroDivisionError".

{pstd}
The command requires three variables. The first variable is the reward. The second is a categorical variable for the assigned treatment arm which is also called "chosen arm". The third variable is a categorical variable for the batch.

{pstd}
The {opt reference_arm} option specifies the reference arm for the inference, defaulting to 0. The {opt test_value} option specifies the test value for the inference, with a default of 0.0. If {opt thompson} is specified, Thompson sampling is used. The {opt size} option sets the sample size for Thompson sampling, defaulting to 100. The {opt clipping} option specifies a clipping value for the inference, defaulting to 0.05. If {opt plot_thompson} is specified, the Thompson sampling results are plotted.

{marker requirements}{...}
{title:Requirements}
{pstd} The underlying calculations are computed in Python. Therefore, python and the respective python packages have to be installed.
At least Stata 16 is required. To calculate the BOLS estimates and confidence intervals, each arm should be played at least once in each batch. 
Otherwise, python will return an error like "ZeroDivisionError".

{marker results}{...}
{title:Stored results}

{pstd} 
{cmd:bbandits} stores the following in {cmd:e()}:

{p2col 5 23 26 2: Scalars}{p_end}

{synoptset 20 tabbed}
{synopt:{cmd:e(N)}} number of observations.{p_end}

{p2col 5 23 26 2: Matrices}{p_end}

{synoptset 20 tabbed}
{synopt:{cmd:e(res)}} matrix with all output results.{p_end}
{synopt:{cmd:e(batch_ols_coefficients)}} Matrix with OLS coefficients for each batch.{p_end}
{synopt:{cmd:e(batched_ols_weights)}} Weight matrix that contains the BOLS weights for each batch.{p_end}

{marker options}{...}
{title:Options}
{phang}{opt r:eference_arm(int 0)} specifies the reference arm for the inference. The default value is 0.

{phang}{opt t:est_value(real 0.0)} specifies the test value for the inference. The default value is 0.0.

{phang}{opt p:lot_thompson} specifies whether to plot the results of Thompson sampling.

{phang}{opt t:est_value(real 0.0)}

{phang}{opt st:acked } specifies whether to plot the stacked plot.

{phang}{opt twooptions_thompson:(string)} takes user-specific two-way options for the twoway Thompson plot.

{phang}{opt twooptions_bols:(string)} takes user-specific two-way options for the plot of the BOLS treatment effects.

{phang}{opt twooptions_ols:(string)} takes user-specific two-way options for the plot of the BOLS and the OLS treatment effects.

{phang}{opt twooptions_sharebybatch:(string)} takes user-specific two-way options for the plot of the shares assigned to each treatment arm by batch.

{phang}{opt twoptions_stackedsharebybatch:(string)} takes user-specific two-way options for the plot of the shares assigned to each treatment arm by batch but stacked as an area.

{phang}{opt twooptions_cumsharesbyybatch:(string)} takes user-specific two-way options for the plot of the cumulative shares assigned to each treatment arm by batch stacked as an area.
  


{marker examples}{...}
{title:Examples}
{hline}
{pstd}{bf:Example 1: Basic Usage}

{phang2}{cmd:. bbandits reward chosen_arm batch}

{pstd}Performs bandit inference on the specified variables.

{phang2}{cmd:. bbandits reward chosen_arm batch, reference_arm(1) test_value(0.5)}

{pstd}Performs bandit inference with a reference arm of 1 and a test value of 0.5.

{hline}
{pstd}
Authors: Jan Kemper, Davud Rostam-Afschar

