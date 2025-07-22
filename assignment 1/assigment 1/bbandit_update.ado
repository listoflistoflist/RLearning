

python:


from bbandit_functions import *

keys = ['Beta_OLS', 'Beta_BOLS_aggregated', 'Z-value', 'P-value', 'CI_lower_bound_95', 'CI_upper_bound_95', 'Treatment_arm_n', 'Reference_arm_n']

end

program bbandit_update
	version 17
	* input reward , chosen_arm, batch, [options]
	syntax varlist [, Thompson Greedy Clipping(real 0.05) Epsilon(real 0.1) Seed(integer 1234) EXcel(string)]
	*** Parse input -> so that chosen_arm is from 0-k in the python function
	* result chosen_arm ordered from 0 to k
	qui capture drop label_chosen_arm
	qui rename `2' chosen_arm
	* Check if it is a string variable
	capture confirm numeric variable `2'
                if !_rc {
                       qui tostring `2', gen(label_chosen_arm) force
					   qui capture drop chosen_arm
					   qui rename label_chosen_arm chosen_arm
                }

	* run inference command in python
	python: reward, chosen_arm, batch  = Data.get(var="`1'"), Data.get(var="chosen_arm"), Data.get(var="`3'")

	* run inference command in python
	python: reward, chosen_arm, batch = np.array(reward), np.array(chosen_arm), np.array(batch) 
	python: reward = reward.astype(float)
	python: batch = batch.astype(int)
	python: df = pd.DataFrame({"reward": reward, "chosen_arm": chosen_arm, "batch": batch})

	* correct nan values
	python: df.loc[(df['chosen_arm'] == ".") | (df['chosen_arm'] == "") , 'chosen_arm'] = np.nan # So that the factorize function works --> cover both missing cases, either from numeric numbers "." or from strings ""
	python: df.loc[df['reward'] == -2147483648 , 'reward'] = np.nan # Because nans are wrongly read in
	python: df['chosen_arm_label'] = df['chosen_arm'] # save the label
	* factorize chosen_arm
	python: df['chosen_arm'] = pd.factorize(df['chosen_arm'])[0] # numeric variable
	*python: print(df)
	python: df.loc[df['chosen_arm'] == -1, 'chosen_arm' ] = np.nan
	
	python: pre = thompson_updating_preprocessing(df)
	python: t1 = pre["df"]
	python: next_batch = pre["next_batch"]
	
	*** Chose algorithm ***
	if "`thompson'" != "" {
		
		display("Thompson sampling algorithm arm probabilities:")
		
		python: probabilities, exact_values = thompson_updating(t1["chosen_arm"].astype(int), t1["reward"], t1["batch"], pre["batch_size"], clipping_rate = `clipping')
		* Display and store results
		python: arm_values = pre["arm_values"]
		python: Matrix.store("e(probabilities)", probabilities)
		python: Matrix.store("e(arm_labels)", arm_values)
		python: arm_values_str = [str(i) for i in arm_values]
		python: Matrix.setRowNames("e(probabilities)", arm_values_str)

		python: final = update_randomization(df, next_batch, probabilities, arm_values)

	} 
	else if "`greedy'" != "" {
		
		display("Epsilon Greedy algorithm arm probabilities:")
		python: shares, chosen_arms, arm_values = epsilon_greedy_updating(t1["chosen_arm"].astype(int), t1["reward"], t1["batch"], pre["batch_size"],  epsilon = `epsilon')
		python: Matrix.store("e(probabilities)", shares)
		python: Matrix.store("e(arm_labels)", arm_values)
		python: final = update_shuffling(df, next_batch, chosen_arms)

	} 
	else {
		* If nothing is specified it should be the greedy algorithm
		display("Epsilon Greedy algorithm arm probabilities:")
		python: shares, chosen_arms, arm_values = epsilon_greedy_updating(t1["chosen_arm"].astype(int), t1["reward"], t1["batch"], pre["batch_size"],  epsilon = `epsilon')
		python: print(shares)
		python: Matrix.store("e(probabilities)", shares)
		python: Matrix.store("e(arm_labels)", arm_values)
		python: final = update_shuffling(df, next_batch, chosen_arms)
	}
	
	* Merge 
	python : unique_pairs = df[['chosen_arm', 'chosen_arm_label']].drop_duplicates() # Merge the original DataFrame with unique_pairs to replace chosen_arm with chosen_arm_label
	python: unique_pairs = unique_pairs.dropna(subset=['chosen_arm_label'])
	python: final = final.drop(columns='chosen_arm_label') # Drop chonsen_arm_label so that the new label can be merged 
	python: final = final.merge(unique_pairs, on='chosen_arm', how='left')
	python: final["chosen_arm_label"] = final["chosen_arm_label"].astype(str)

	
	*** Initialize variables in stata
	capture drop chosen_arm 
	capture drop reward 
	capture drop batch
	capture drop chosen_arm_numeric
	qui gen reward = .
	qui gen chosen_arm = ""
	qui gen batch = .
	qui gen chosen_arm_numeric = .
	
	python: Data.store("reward", None, final['reward'], None)
	python: Data.store("chosen_arm", None, final['chosen_arm_label'], None)
	python: Data.store("batch", None, final['batch'], None)
	python: Data.store("chosen_arm_numeric", None, final['chosen_arm'], None)
	
	* fix "nan" should be come .
	qui replace chosen_arm = "" if chosen_arm == "nan"

	* Get scalar value
	python: current_batch = next_batch-1
	python: Scalar.setValue('e(current_batch)', current_batch)
	
	* format returns
		*** Update
	* Display results nicely:
	tempname value name
	mat `value' = e(probabilities)
	mat `name' = e(arm_labels)
	local rows = rowsof(`name')
	* Display the header
	   display "Arm Label Numeric        Probability"
	forval i = 1/`rows' {
		local name1 = `name'[`i', 1]
        local value1 = `value'[`i', 1]
        display "Arm `name1'            = `value1'"
		
	}
		
   * If `excel' option is provided, proceed with exporting to Excel
    if "`excel'" != "" {

        local path "`excel'"
        display "Dataset is exported to: `path'"
        display "`e(current_batch)'"
        local date : di  %tdCY-N-D  daily("$S_DATE", "DMY")

        * Create the full path including the file name
        local fullpath "`path'/`date'_bandit_experiment_batch=`e(current_batch)'.xlsx"

        * Export to Excel
        export excel using "`fullpath'", firstrow(variables)
    }
	
end	





