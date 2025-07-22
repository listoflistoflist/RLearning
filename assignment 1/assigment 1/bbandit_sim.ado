
*** Read in python files and define functions

python:

from bbandit_functions import *

end


*** stata subroutines

* epsilon greedy 
program e_greedy, eclass
	
	/*
	 Implementation of epsilon greedy algorithm.

	Parameters:
	  anything (str): list of float (true mean values)
      Batch (int): Batch size.
      Size (int): Size of input.
      Eps (float): Epsilon value.
  
	 Returns:
	 Python:
      chosen_arm_list (list): List of chosen arms.
      rewards_list (list): List of rewards.
      batch_indicator_list (list): List of batch indicators
	 Stata:
		Results matrix
	*/
	
	syntax anything [, Batch(int 25) Size(int 100) Eps(real 0.1) Exploration(int 0) Decay(real 1)]
		
	*display `anything'
	capture matrix drop greedy
	* Python implementation
	python: list_of_true_mean = `anything' # Read in anything
	python: list_of_true_mean = list_of_true_mean.split() # Split based on tabs
	python: list_of_true_mean = np.array(list_of_true_mean, dtype = float) # save as float array
	*python: print(list_of_true_mean)
	
	*python: t = np.array([`batch', `size', `eps'])
	*python: print(t)
	python: results = greedy_alg_k(list_of_true_mean, `eps', `batch', `size', `exploration', `decay')
	python: keys = ["chosen_arm_list", "rewards_list", "batch_indicator_list"]
	python: values = [results.get(key) for key in keys]
	python: mat_res = np.array(values).T
	
	* save as a matrix in back into stata
	python: Matrix.store("greedy", mat_res)
	
	* Save other outputs as estimation results
	python: decay_res = results["decay_rate"]
	
	python: Matrix.store("decay_res", decay_res)
	* transform into stata dataset
	clear
	qui svmat greedy, names(col)
	rename (c1 c2 c3) (chosen_arm reward batch)
	
	* return estimates
	ereturn matrix decay_rate = decay_res
	
end

* bernoulli thompson

program thompson_bernoulli, eclass

/*
	Function Name: thompson_bernoulli

	Description:
	This Stata program implements the Thompson Sampling algorithm for the Bernoulli distribution. It takes as input a list of true mean probabilities for each arm of a multi-armed bandit problem. The program then runs the Thompson Sampling algorithm in Python, which returns the chosen arm for each round, the corresponding rewards, and batch indicators. Additionally, it retrieves the alpha and beta values used in the Thompson Sampling algorithm and stores them as matrices in Stata.

	Syntax:
	thompson_bernoulli anything [, Batch(int 25) Size(int 100) Clipping(real 0.05) Exploration(int 0)]

	Parameters:
	- anything (str): Input data containing the true mean probabilities for each arm, separated by spaces.
	- Batch (int): Batch size (default is 25).
	- Size (int): Size of input (default is 100).
	- Clipping (real): Clipping parameter for Thompson Sampling (default is 0.05).
	- Exploration (int): Exploration parameter for Thompson Sampling (default is 0).

	Returns:
	Python:
	- chosen_arm (int): List of chosen arms for each round.
	- reward (int): List of rewards obtained for each chosen arm.
	- batch (int): List of batch indicators.
	Stata:
	- matrix with results
	- matrix with alpha values
	- matrix with beta values
	- Plot of beta distribution after each round
*/

	version 17
	syntax anything [, Batch(int 25) Size(int 100) Clipping(real 0.05) Exploration(int 0) Decay(real 1) Plot_thompson STacked TWoptions(string asis)]
	*drop resulting matrices - otherwise problems when dimensions of outcome matrix changes
	capture matrix drop beta_list
	capture matrix drop alpha_list  
	capture matrix drop thompson
	capture matrix drop clipping_rate
	capture matrix drop true
	* python implementation
	python: list_of_true_mean = `anything'
	python: list_of_true_mean = list_of_true_mean.split()
	python: list_of_true_mean = np.array(list_of_true_mean, dtype = float)
	* run epsilon greedy algorithm from python
	python: results = bernoulli_thompson_batched_clipping(list_of_true_mean, `batch', `size', `clipping', `exploration', `decay')
	python: keys = ["chosen_arm_list", "rewards_list", "batch_indicator_list"]
	python: values = [results.get(key) for key in keys]
	python: mat_res = np.array(values).T
	* Get alpha and beta values into stata
	python: keys = ["alpha_values_list", "beta_values_list"]
	python: values = [results.get(key) for key in keys]
	python: mat_alpha = np.array(values[0])
	python: mat_beta = np.array(values[1])

	* save as a matrix in back into stata
	python: Matrix.store("true", list_of_true_mean)
	python: Matrix.store("thompson", mat_res)
	python: Matrix.store("alpha_list", mat_alpha)
	python: Matrix.store("beta_list", mat_beta)
	* Save other outputs as estimation results
	python: clipping_rate = results["clipping_rate_list"]
	
	python: Matrix.store("clipping_rate", clipping_rate)
	ereturn matrix decay_rate = clipping_rate
* transform into stata dataset
	clear
	qui svmat thompson, names(col)
	rename (c1 c2 c3) (chosen_arm reward batch)	

di `twoptions'
	
	if "`plot_thompson'" == "" & "`stacked'" != ""{
	di in red "please use stacked with plot_thompson. That is specify as option:" in ye " plot_thompson stacked"	
	}
	
	if "`plot_thompson'" != ""{
	* get batch size 
	local batch_size = rowsof(beta_list)
	local arms = colsof(beta_list) 
	di "Number of arms: " + `arms'
	di "Number of batches: " + `batch_size'
	*

		forvalues i =1/`batch_size' {
					local name =  "t" + "`i'"
					di "`name'"
					if "`stacked'" == ""{
						local vertical_line
						local beta_densities
						local legend_label
						forv j=1/`=`arms'' {
							local beta_densities `beta_densities' (function y=betaden(alpha_list[`i', `j'], beta_list[`i', `j'] , x), range(0 1)) 
							local vertical_line `vertical_line' xline(`=true[`j',1]', lpattern(dash) lc(gray) lwidth(medthick))
							local legend_label `legend_label' label(`j' "Arm `j'")
									
						   }
						   
						twoway `beta_densities' ///
							   , ///
							   `vertical_line' ///
								name("`name'", replace) legend( `legend_label') `=`twoptions''

					}
					if "`stacked'" != ""{
				local combine
				forv j=1/`=`arms'' {

							tw (function y = betaden(alpha_list[`i', `j'], beta_list[`i', `j'],x), range(0.01 0.99) lwidth(medthick)),  xline(`=true[`j',1]', lpattern(dash) lc(black) lwidth(medthick)) ///
						   ytitle("B(`=alpha_list[`i', `j']',`=beta_list[`i', `j']')" "density") ylabel(#0, nolabels nogrid) xlabel(, nogrid) xtitle(Share of successes) plotr(m(zero)) name("g_`j'_`i'", replace) nodraw `=`twoptions'' /**/ 
							local combine "`combine' g_`j'_`i'"
					   }
						gr combine `combine', xcommon col(1) iscale(1) name("combine_`i'", replace)
						}
					
				}
	}
	
	
	
end 



program epsilon_greedy_simulation
	version 17
	syntax anything [, Batch(int 25) Size(int 100) Eps(real 0.1) N(int 1000) Decay(real 1) Reference_arm(int 0), Arm(int 1), Test_value(real 0.0) Exploration(int 0) Plot TWoptions(string asis)]
	capture matrix drop	greedy_simulation
	python: list_of_true_mean = `anything'
	python: list_of_true_mean = list_of_true_mean.split()
	python: list_of_true_mean = np.array(list_of_true_mean, dtype = float)
	* run epsilon greedy algorithm from python
	python: results = simulation_greedy(`n' ,list_of_true_mean, `eps', `batch', `size', `exploration', `decay', `reference_arm', `arm', `test_value')
	python: keys = ["beta_test_statistic", "bols_test_statistic"]
	python: values = [results.get(key) for key in keys]
	python: mat_res = np.array(values).T
	* save as a matrix in back into stata
	python: Matrix.store("greedy_simulation", mat_res)

* transform into stata dataset
clear
qui svmat greedy_simulation, names(col)
rename (c1 c2) (beta_test_statistic bols_test_statistic)

tokenize `anything'
local k : display %04.3f `1'
local l : display %04.3f `2'
if "`plot'" != ""{

	tw (hist beta_test_statistic if beta_test_statistic>=`=`test_value'-4' & beta_test_statistic<=`=`test_value'+4', width(0.1)) ///
	(kdensity beta_test_statistic if beta_test_statistic>=`=`test_value'-4' & beta_test_statistic<=`=`test_value'+4') ///
	(function y=normalden(x, `test_value', 1) , range(`=`test_value'-4' `=`test_value'+4')) ///
	, legend(off) name(OLS, replace) xtitle("OLS test statistic with true values " "`k', `l'" " batch size=`=`size'' with `=`batch'' batches, repetitions=`=`n''")  `=`twoptions'' xline(`test_value', lp(dash) lc(gs6))
	
	tw (hist bols_test_statistic if bols_test_statistic>=`=`test_value'-4' & bols_test_statistic<=`=`test_value'+4', width(0.1)) ///
	(kdensity bols_test_statistic if bols_test_statistic>=`=`test_value'-4' & bols_test_statistic<=`=`test_value'+4') ///
	(function y=normalden(x, `test_value', 1) , range(`=`test_value'-4' `=`test_value'+4')) ///
	, legend(off) name(BOLS, replace) xtitle("BOLS test statistic with true values " "`k', `l'" " batch size=`=`size'' with `=`batch'' batches, repetitions=`=`n''")  `=`twoptions'' xline(`test_value', lp(dash) lc(gs6))

}

end



program bernoulli_thompson_simulation
	version 17
	syntax anything [, Batch(int 25) Size(int 100) Clipping(real 0.05) Test_value(real 0.0) Reference_arm(int 0) Arm(int 1) Exploration(int 0) N(int 1000) Plot TWoptions(string asis)] // Here clipping noch einführen
	* drop resulting matrix
	capture matrix drop thompson_simulation
	* python
	python: list_of_true_mean = `anything'
	python: list_of_true_mean = list_of_true_mean.split()
	python: list_of_true_mean = np.array(list_of_true_mean, dtype = float)
	* run epsilon greedy algorithm from python
	python: results = simulation_thompson(`n' ,list_of_true_mean, `batch', `size', `clipping', `exploration', `reference_arm', `arm', `test_value')
	python: keys = ["beta_test_statistic", "bols_test_statistic"]
	python: values = [results.get(key) for key in keys]
	python: mat_res = np.array(values).T
	* save as a matrix in back into stata
	python: Matrix.store("thompson_simulation", mat_res)

* transform into stata dataset
clear
qui svmat thompson_simulation, names(col)
rename (c1 c2) (beta_test_statistic bols_test_statistic)

tokenize `anything'
local k : display %04.3f `1'
local l : display %04.3f `2'
if "`plot'" != ""{

	tw (hist beta_test_statistic if beta_test_statistic>=`=`test_value'-4' & beta_test_statistic<=`=`test_value'+4', width(0.1)) ///
	(kdensity beta_test_statistic if beta_test_statistic>=`=`test_value'-4' & beta_test_statistic<=`=`test_value'+4') ///
	(function y=normalden(x, `test_value', 1) , range(`=`test_value'-4' `=`test_value'+4')) ///
	, legend(off) name(OLS, replace) xtitle("OLS test statistic with true values " "`k', `l'" " batch size=`=`size'' with `=`batch'' batches, repetitions=`=`n''")  `=`twoptions'' xline(`test_value', lp(dash) lc(gs6))
	
	tw (hist bols_test_statistic if bols_test_statistic>=`=`test_value'-4' & bols_test_statistic<=`=`test_value'+4', width(0.1)) ///
	(kdensity bols_test_statistic if bols_test_statistic>=`=`test_value'-4' & bols_test_statistic<=`=`test_value'+4') ///
	(function y=normalden(x, `test_value', 1) , range(`=`test_value'-4' `=`test_value'+4')) ///
	, legend(off) name(BOLS, replace) xtitle("BOLS test statistic with true values " "`k', `l'" " batch size=`=`size'' with `=`batch'' batches, repetitions=`=`n''")  `=`twoptions'' xline(`test_value', lp(dash) lc(gs6))

}

end





program bbandit_sim

version 17
syntax anything [, Batch(int 25) Size(int 100) Eps(real 0.1) Clipping(real 0.05) EXploration_phase(int 0) Test_value(real 0.0) Greedy Thompson Monte_carlo N(int 1000) Decay(real 1) Reference_arm(int 0) Arm(int 1) Plot_thompson STacked TWopts(string asis)]
	
if ("`greedy'" != "" & "`monte_carlo'" == "") | ("`greedy'" == "" & "`thompson'" == "" ){ /// Default value epsilon greedy
	
	display "Epsilon greedy"
	
	e_greedy "`anything'", batch(`batch') size(`size') eps(`eps') decay(`decay')

	}

if "`thompson'" != "" & "`monte_carlo'" == "" {
	
	*** detect syntax errors
	* Input greater than 1
	 foreach var of local anything {
        // Check if the current input value is greater than 1
        if `var' > 1 {
            // Display an error message
            di as error "Error: Input value `" `var' "' is greater than 1. The Bernoulli Thompson algorithm only permits valid probabilities ([0,1])"
            // Exit the program
            exit
        }
    }
	
	display "Thompson algorithm"
	thompson_bernoulli "`anything'", batch(`batch') size(`size') clipping(`clipping') exploration(`exploration_phase') decay(`decay') `plot_thompson' `stacked' tw(`"`twopts'"')
}	

*** Monte Carlo 
* Epsilon greedy 
if "`monte_carlo'" != "" & "`greedy'" != "" {
	
	display "Epsilon greedy - Monte Carlo simulation"
	epsilon_greedy_simulation "`anything'", batch(`batch') size(`size') eps(`eps') test_value(`test_value') reference_arm(`reference_arm') arm(`arm') n(`n') exploration(`exploration_phase') plot tw(`"`twopts'"') /// Always plot
	
}

if "`monte_carlo'" != "" & "`thompson'" != "" {
	
	display "Bernoulli thompson sampling - Monte Carlo simulation"
	bernoulli_thompson_simulation "`anything'", batch(`batch') size(`size') clipping(`clipping') test_value(`test_value') reference_arm(`reference_arm') arm(`arm') n(`n') plot tw(`"`twopts'"') /// Always plot
	
}

end



