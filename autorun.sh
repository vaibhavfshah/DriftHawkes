# Run the Python script with seeds 43, 44, and 45 for each parameter set

string_param_synth=('4	128	8	8	8	2	2	8	0.01	200	64	20')
dataset_param_synth=('synthetic,synthhawkes')

string_param_sl=('2	512	16	8	16	1	5	25	0.01	200	64	20')
dataset_param_sl=('yelp,saint louis')

string_param_tampa=('2	1024	8	8	16	1	5	25	0.01	200	64	20')
dataset_param_tampa=('yelp,tampa')

string_param_reno=('2	1024	4	4	16	1	5	25	0.01	200	64	20')
dataset_param_reno=('yelp,reno')

string_param_tucson=('2	512	2	2	16	1	5	25	0.01	200	64	20')
dataset_param_tucson=('yelp,tucson')

#change param
string_param=$string_param_sl
dataset_param=$dataset_param_sl

for int_param in 43 44 45; do
    echo "Running: python run.py --arg1 '$int_param' --arg2 '$string_param' --arg3 '$dataset_param'"
    python run.py --arg1 "$int_param" --arg2 "$string_param" --arg3 "$dataset_param"
done
