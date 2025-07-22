*! bbandits, version 1, 04.09.2024
*! Authors: Jan Kemper, Davud Rostam-Afschar


python:


from bbandit_functions import *

keys = ['Beta_OLS', 'Beta_BOLS_aggregated', 'Z-value', 'P-value', 'CI_lower_bound_95', 'CI_upper_bound_95', 'Treatment_arm_n', 'Reference_arm_n']

end



program bbandits, eclass
	version 17
	syntax varlist [if] [in] [, Reference_arm(int 0) Test_value(real 0.0) Plot_thompson STacked twoptions_thompson(string asis) twoptions_ols(string asis) twoptions_bols(string asis) twoptions_sharebybatch(string asis) twoptions_stackedsharebybatch(string asis) twoptions_cumsharesbyybatch(string asis)] 
	
	capture matrix drop alpha_list beta_list

*** Inference
	* pre-process data in stata
	* result chosen_arm ordered from 0 to k
	capture drop label_chosen_arm
	capture drop chosen_arm_py
	* create a tempvar for chosen_arm
	tempvar chosen_arm_py	
	* Check if it is a string variable
	capture confirm numeric variable `2'
                if !_rc {
                       qui tostring `2', gen(label_chosen_arm) force
                }
	capture gen label_chosen_arm = `2'
	encode label_chosen_arm, gen(label_chosen_fac)
	gen `chosen_arm_py' = label_chosen_fac
	label values `chosen_arm_py'
	qui replace `chosen_arm_py' = `chosen_arm_py' - 1
	drop label_chosen_fac 
	capture confirm variable chosen_arm
	if !_rc {
		// The variable exists, do something
		drop chosen_arm // Not an elegant solution to drop the existing chosen_arm, better to work with tempvars
		gen chosen_arm = `chosen_arm_py'
	}
	else {
		// The variable does not exist, do something else or nothing
		gen chosen_arm = `chosen_arm_py'
	}
	* run inference command in python
	python: reward, chosen_arm, batch  = Data.get(var="`1'"), Data.get(var="chosen_arm"), Data.get(var="`3'")
	python: data = pd.DataFrame({"reward": reward, "chosen_arm": chosen_arm, "batch": batch})
	python: results, ereturn_results = bols_inference_k(data, chosen_arm = "chosen_arm", reward = "reward", batch = "batch", reference_arm = `reference_arm', test_value = `test_value')
	* save results to matrix
	python: values = [results.get(key) for key in keys]
	python: mat_res = np.array(values).T 
	
	python: batch_beta = ereturn_results["batch_beta"]
	python: weights_bols = ereturn_results["weights_bols"]
	
	* save beta coefficients for each batch
	python: Matrix.store("adaptive_inference", mat_res)
	python: Matrix.setColNames("adaptive_inference", keys)
	
	* beta bols estimates
	python: Matrix.store("batch_beta", batch_beta)
	python: Matrix.store("weights_bols", weights_bols)
	
	*matrix list weights_bols
	ereturn matrix res adaptive_inference 
	ereturn matrix batch_ols_coefficients batch_beta
	ereturn matrix batched_ols_weights weights_bols
	
	tempname res batch_ols_coefficients batched_ols_weights
	mat `res' = e(res) 
	mat `batch_ols_coefficients' = e(batch_ols_coefficients)
	mat `batched_ols_weights' = e(batched_ols_weights)
	
	* store estimation results
	* Display the results in a nice fashion
	*** Plot thompson development if specified
	if "`plot_thompson'" == "" & "`stacked'" != ""{
	di in red "please use stacked with plot_thompson. That is specify as option:" in ye " plot_thompson stacked"	
	}
	
	if "`plot_thompson'" != "" {
		
	display "Distribution by batch for Thompson sampling"
		* Get alpha and beta values if it is a thompson algorithm
	python: ts_list = get_alpha_beta(reward, chosen_arm, batch)
		* store values for thompson
	python: Matrix.store("alpha_list", ts_list["alpha_list_cum"])
	python: Matrix.store("beta_list", ts_list["beta_list_cum"])
	
	local batch_size = rowsof(beta_list)
	local arms = colsof(beta_list) 
	di "Number of arms: " + `arms'
	di "Number of batches: " + `batch_size'
				 
		forvalues i =1/`batch_size' {
					local name =  "t" + "`i'"
					di "`name'"

					
					if "`stacked'" == ""{
						local beta_densities
						local legend_label
						forv j=1/`=`arms'' {
							local beta_densities `beta_densities' (function y=betaden(alpha_list[`i', `j'], beta_list[`i', `j'] , x), range(0 1)) 
							local legend_label `legend_label' label(`j' "Arm `j'")
									
						   }
						   
						twoway `beta_densities' ///
							   , ///
								name("`name'", replace) legend( `legend_label') `=`"`twoptions_thompson'"''

					}
					if "`stacked'" != ""{
				local combine
				forv j=1/`=`arms'' {

							tw (function y = betaden(alpha_list[`i', `j'], beta_list[`i', `j'],x), range(0.01 0.99) lwidth(medthick)),  ///
						   ytitle("B(`=round(alpha_list[`i', `j'])',`=round(beta_list[`i', `j'])')" "density") ylabel(#0, nolabels nogrid) xlabel(, nogrid) xtitle(Share of successes) plotr(m(zero)) name("g_`j'_`i'", replace) nodraw  `=`"`twoptions_thompson'"'' /**/ 
							local combine "`combine' g_`j'_`i'"
					   }
						gr combine `combine', xcommon col(1) iscale(1) name("combine_`i'", replace)
						}
					
				}
 
}

*** Histogram shares ***	
	tempname res armnr share share2 N fixed max rewards rewards_bandits rewards_balanced // temp variable leads to not returning the stata estimates
	
	mat `res' = e(res)
	mat `armnr' = .
	
qui tab chosen_arm, matcell(`share')
	mat `rewards' = `res'[1...,1]
	
	mata : st_matrix("`N'", colsum(st_matrix("`share'")))
	
	mat `fixed' = `N'[1,1]*`=1/(`=rowsof(`res')+1')'

	mata : st_matrix("`max'", colmax(st_matrix("`rewards'")))
	mata : st_matrix("`share'", st_matrix("`share'")/colsum(st_matrix("`share'")))

qui levelsof chosen_arm

qui hist chosen_arm , discrete frac xtitle(Treatment Arm) xlabel(0(1)`=max(`=subinstr("`r(levels)'"," ",",",.)')', valuelabel angle(90)) ylabel(#10) xlabel(,nogrid) ytitle("Share of Arm Selected") name("ShareArmSelected", replace)

*** Main table ***
		qui su `1' if chosen_arm ==`reference_arm' 
			mat  `rewards'=`rewards'+J(`=rowsof(`res')', 1, r(mean))

		mata : st_matrix("`rewards_balanced'", st_matrix("`rewards'"):*st_matrix("`fixed'"))
		
	
	mat `share2'=`share'[2...,1]

		mata : st_matrix("`rewards'", st_matrix("`rewards'"):*st_matrix("`share2'"):*st_matrix("`N'"))
	    mata : st_matrix("`rewards_bandits'", colsum(st_matrix("`rewards'")))
	    mata : st_matrix("`rewards_balanced'", colsum(st_matrix("`rewards_balanced'")))

	di _n _skip(18) 	
		di in smcl in gr %9s _skip(-2) "Number of obs                       =  "  in ye %10.0f `N'[1,1]
		di in smcl in gr %9s _skip(-2) "Est. Rewards only best arm          =  "  in ye %10.0f `=(`max'[1,1]+r(mean))*`N'[1,1]' in gr %9s _skip(4) 								  "Mean reward best arm      =  "  in ye %10.4f `=(`max'[1,1]+r(mean))*`N'[1,1]/`N'[1,1]'
		di in smcl in gr %9s _skip(-2) "Actual total reward                 =  "  in ye %10.0f `=round(`rewards_bandits'[1,1]+`share'[1,1]*`N'[1,1]*r(mean))'  in gr %9s _skip(4) "Actual mean reward        =  "  in ye %10.4f `=round(`rewards_bandits'[1,1]+`share'[1,1]*`N'[1,1]*r(mean))/`N'[1,1]'
		di in smcl in gr %9s _skip(-2) "Est. reward uniformly chosen arms   =  "  in ye %10.0f `=round(`rewards_balanced'[1,1]+`fixed'[1,1]*r(mean))'   in gr %9s _skip(4)		  "Mean reward uniform       =  "  in ye %10.4f `=round(`rewards_balanced'[1,1]+`fixed'[1,1]*r(mean))/`N'[1,1]'
		di in smcl in gr "{hline 10}{c TT}{hline 13}{c TT}{hline 72}"
	    di in smcl in gr %9s _skip(-2) "Arm b" _skip(1) "{c |} Mean Reward {c |} " in gr _skip(58)  "Share arm b"
		di in smcl in gr "{hline 10}{c +}{hline 13}{c +}{hline 72}"



		di in smcl _skip(4) in gr "`num_str' " in ye 	/*
		*/ _skip(4) in gr " {c |} " in ye _skip(0) %10.4f r(mean)  _skip(1)  in gr " {c |} " _skip(55) in ye  %10.4f `share'[1,1]  

				

		di in smcl in gr "{hline 10}{c +}{hline 13}{c +}{hline 72}"
	    di in smcl in gr %9s _skip(-2) "k v. b" _skip(1) "{c |}  Margin OLS {c |}"  _skip(3) "Margin BOLS" _skip(7) "z" _skip(7) "P>|z|" _skip(3) "[95% Conf. Interval]" _skip(2) "Share arm k" _skip(3)		
		di in smcl in gr "{hline 10}{c +}{hline 13}{c +}{hline 72}"
		forvalues i = 1/`=rowsof(`res')' {
	    local num_str : di %8s "`i'-0"
		di in smcl _skip(0) in gr "`num_str' " in ye 	/*
		*/ _skip(0) in gr " {c |} " in ye _skip(0) %10.4f `res'[`i',1]  _skip(1)  in gr " {c |} " _skip(1) in ye  %10.4f `res'[`i',2]  _skip(1) in ye  %10.2f `res'[`i',3]  _skip(1) in ye  %10.3f `res'[`i',4]  		_skip(1) in ye  %10.4f `res'[`i',5]  _skip(1) in ye  %10.4f `res'[`i',6]  _skip(0) in ye  %10.4f `share'[`=`i'+1',1] 
		mat `armnr' = `armnr' \ `i'
		}
		di in smcl in gr "{hline 10}{c BT}{hline 13}{c BT}{hline 72}"

		matrix `armnr' = `armnr'[2..`=rowsof(`res')+1', 1]
	mat `res' = `res' , `armnr'

	mata : st_matrix("`res'", sort(st_matrix("`res'"), 2))

*** BOLS graph ***
	tempname bols ols ci_l ci_u arm

	qui gen `bols' = `res'[_n,2] in 1/`=rowsof(`res')'
	qui gen `ols' = `res'[_n,1] in 1/`=rowsof(`res')'
	qui gen `ci_l' = `res'[_n,5] in 1/`=rowsof(`res')'
	qui gen `ci_u' = `res'[_n,6] in 1/`=rowsof(`res')'
	qui gen `arm' = _n in 1/`=rowsof(`res')'
	scalar best = `=`res'[`=rowsof(`res')',9]' 
	scalar worst = `=`res'[1,9]'

	local xlab ""
		forvalues i = 1/`=rowsof(`res')' {
			local xlab "`xlab' `i' "`""`=`res'[`i',9]'""'" "
		}
		local x: di "`xlab'"

twoway (scatter `bols' `arm', mcolor("142 69 97")) ///
       (rcap `ci_l' `ci_u' `arm', lcolor("142 69 97")) ///
       , legend(off ) yline(0, lpattern(dash) lcolor("85 164 168")) ///
       xlabel(`x', noticks) ylabel(#6, grid) ///
       xtitle("Arm") ytitle("Treatment Effect") plotregion(margin(b=0)) name("BOLS", replace) ///
	   `=`"`twoptions_bols'"'' 

*** OLS graph ***
 qui reg `1' i.chosen_arm
 	tempname ols_ci_l ols_ci_u arm se

	mata : st_matrix("`se'", sqrt(diagonal(st_matrix("e(V)"))))

local alpha = 0.05
	
 mat `ols_ci_l' = e(b)' - invttail(e(N)-1,0.5*`alpha')*`se' 
 mat `ols_ci_u' = e(b)' + invttail(e(N)-1,0.5*`alpha')*`se'

	mata : st_matrix("`res'", sort(st_matrix("`res'"), 9))
	
		mat `res' = `res' , `ols_ci_l'[2..`=rowsof(`ols_ci_l')-1', 1], `ols_ci_u'[2..`=rowsof(`ols_ci_u')-1', 1]

	tempname ci_l_ols ci_u_ols arm2

	mata : st_matrix("`res'", sort(st_matrix("`res'"), 2))

	qui gen `ci_l_ols' = `res'[_n,10] in 1/`=rowsof(`res')'
	qui gen `ci_u_ols' = `res'[_n,11] in 1/`=rowsof(`res')'
	qui gen `arm' = _n in 1/`=rowsof(`res')'
	qui gen `arm2' = `arm'+0.2 in 1/`=rowsof(`res')'

twoway (scatter `ols' `arm2', mcolor(gs13) m(D)) ///
       (rcap `ci_l_ols' `ci_u_ols' `arm2', lcolor(gs13)) ///
	   (scatter `bols' `arm', mcolor("142 69 97")) ///
       (rcap `ci_l' `ci_u' `arm', lcolor("142 69 97")) ///
       , legend(order(1 3) label(1 "OLS") label(3 "BOLS") ) yline(0, lpattern(dash) lcolor("85 164 168")) ///
       xlabel(`x', noticks) ylabel(#6, grid) ///
       xtitle("Arm") ytitle("Treatment Effect") plotregion(margin(b=0)) name("OLS", replace) ///
	    `=`"`twoptions_ols'"''

*** Share by batch graph ***
tempname all total share cumshare batch_enc
 
capture confirm numeric variable `3'
                if !_rc {
					gen `batch_enc'=`3'
                }
                else {
					gen `batch_enc' = 0
					replace `batch_enc' = 1 if `3'[_n]~=`3'[_n-1]				
					replace `batch_enc' = sum(`batch_enc')	
                }
 
gen `all'=1
preserve
collapse (mean) `1' (count) `all' , by(`batch_enc' chosen_arm)
bys `batch_enc': egen `total'=total(`all')
gen `share'= `all'/`total'
bys `batch_enc': gen `cumshare' = sum(`share')
																																													
local area ""
local line ""
	forvalues i = `=rowsof(`res')'(-1)0 {
		local area "`area' (area `cumshare' `batch_enc' if chosen_arm ==`i', fintensity(100))"
		local line "`line' (line `share' `batch_enc' if chosen_arm ==`i')"
	}
	
qui levelsof `batch_enc' 

qui sum `batch_enc'
tw `line', legend(off) ytitle(Share) xtitle(Batch) ylabel(0(.2)1) xlabel(`r(min)'(1)`r(max)') name("ShareByBatch", replace) ///
`=`"`twoptions_sharebybatch'"''

*** Stacked share by batch graph ***
qui sum `batch_enc'
tw `area', legend(off) ytitle(Share) xtitle(Batch) ylabel(0(.2)1) xlabel(`r(min)'(1)`r(max)') name("StackedShareByBatch", replace) ///
`=`"`twoptions_stackedsharebybatch'"''

*** Cumulative share by batch graph ***
tempname cumall cumttotal cumshare cumshare2

sort chosen_arm `batch_enc'
by chosen_arm: gen `cumall' = sum(`all')
bys `batch_enc' : egen `cumttotal'=total(`cumall')


gen `cumshare'= `cumall'/`cumttotal'
sort `batch_enc' chosen_arm 
by `batch_enc' : gen `cumshare2' = sum(`cumshare')

	
local area ""
	forvalues i = `=rowsof(`res')'(-1)0 {
		local area "`area' (area `cumshare2' `batch_enc' if chosen_arm ==`i', fintensity(100))"
	}
	
qui levelsof `batch_enc'
tokenize "`r(levels)'"
tw `area', legend(off) ytitle(Share) xtitle(Batch) ylabel(0(.2)1) xlabel(`1'(1)`=`1'+r(r)-1') name("CumSharesByBatch", replace) ///
`=`"`twoptions_cumsharesbyybatch'"''

*** restore the main results ***
// Assign new column names to the matrix 'res'
matrix colnames `res' = "margin OLS" "margin OLS BOLS" "z" "p-value" ///
                      "95% BOLS conf lower bound"   "95% conf upper bound" "obs reference arm" ///
                      "obs treatment arm" "treatment arm indicator" ///
                      "95% OLS conf lower bound" "95% OLS conf upper bound"

* Return results
ereturn matrix res `res' 
ereturn matrix batch_ols_coefficients `batch_ols_coefficients'
ereturn matrix batched_ols_weights `batched_ols_weights'


end

