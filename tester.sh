# Example for launching training and evaluation

# test training
python -u /home/abadea/analysis/Pheno/cannonball-rpv-stops/run/train.py -a colalola -i /eos/home-a/abadea/data/rpv_multijet/madgraph/MG5_aMC_v2_7_3/pp_t1t1j_msu3_scan/npz/ptsmear_with_pz_recalculate_with_pt_sampleTillPositive_with_e_recalculate_at_fixed_parton_mass/0.0/shuf1/run_10_1389132_ptsmear0.00_shuf1.npz -o weights.npz -e 1 -b 10000 -j 4 -ns 2 2 -ni 1 -nc 30 -hd 200 200 200 -epe 1e-6 -epwe 1e-6 -epl 5 -epp 30

# test model evaluation
python run/evaluate.py -d cpu -b 40000 -mp 1 -nc 30 -hd 200 200 200 -i /eos/home-a/abadea/data/rpv_multijet/madgraph/MG5_aMC_v2_7_3/pp_t1t1j_msu3_scan/npz/ptsmear_with_pz_recalculate_with_pt_sampleTillPositive_with_e_recalculate_at_fixed_parton_mass/0.0/shuf1/run_10_1389132_ptsmear0.00_shuf1.npz -o ./ -w /hdd01/abadea/analysis/rpvmj/stops/model_final/ptsmear0.0/weights.npz

# test benchmark evaluation on masym
python run/evaluate_benchmark.py -d cpu -i /eos/home-a/abadea/data/rpv_multijet/madgraph/MG5_aMC_v2_7_3/pp_45j/gridpackruns/npz/ptsmear0.2/run_01_2293524_ptsmear0.20_shuf0.npz -o ./ -b 10000 -ni 5 -np 2 -nc 2 -bg -m masym

# test benchmark evaluation on drsum c=1
python run/evaluate_benchmark.py -d cpu -i /eos/home-a/abadea/data/rpv_multijet/madgraph/MG5_aMC_v2_7_3/pp_45j/gridpackruns/npz/ptsmear0.2/run_01_2293524_ptsmear0.20_shuf0.npz -o ./ -b 10000 -ni 5 -np 2 -nc 2 -bg -m drsum -c 1
