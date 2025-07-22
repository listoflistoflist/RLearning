

python:


from bbandit_functions import *

keys = ['Beta_OLS', 'Beta_BOLS_aggregated', 'Z-value', 'P-value', 'CI_lower_bound_95', 'CI_upper_bound_95', 'Treatment_arm_n', 'Reference_arm_n']

end


program bbandit_initialize
version 17
syntax [, Batches(int 3) Arms(int 2) Exploration_phase(int 1)]

python: df = Data.getAsDict()
python: df = pd.DataFrame(df)
python: print(df)
python: df = intialize(df, batches= `batches', arms = `arms', exploration = `exploration_phase')

*** Initialize variables in stata
capture drop chosen_arm 
capture drop reward 
capture drop batch
gen reward = .
gen chosen_arm = .
gen batch = .

python: Data.store("reward", None, df['reward'], None)
python: Data.store("chosen_arm", None, df['chosen_arm'], None)
python: Data.store("batch", None, df['batch'], None)



end