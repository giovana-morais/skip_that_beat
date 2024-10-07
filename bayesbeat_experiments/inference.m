function Results = test_model(model_file, test_files, out_folder)
	% [Results] = test_bayesbeat(in_file, out_folder)
	%   Test bayesbeat.
	% ----------------------------------------------------------------------
	% INPUT Parameter:
	%   in_file             : audio file
	%   out_folder          : folder, where output (beats and model) are stored
	%
	% OUTPUT Parameter:
	%   Results             : structure array with beat tracking results
	%
	% 03.09.2024 by Giovana Morais
	% ----------------------------------------------------------------------

	Params.model_fln = model_file;
	Params.testLab = test_files;
	Params.data_path = out_folder;
	Params.results_path = out_folder;
	Params.save_beats = 1;
	Params.save_downbeats = 1;
	Params.save_rhythm = 1;
	Params.save_meter = 1;
	Params.inferenceMethod = 'HMM_viterbi';

	BT = BeatTracker(Params);

	% run the model for all test files and save the results
	for i = 1:numel(test_files)
		Results = BT.do_inference(i);
		% save results
		[~, fname, ~] = fileparts(Params.testLab{i});

		BT.save_results(Results, out_folder, fname);
	end
end


% Specifies the root folder of the bayes_beat package.
base_path = '/home/gigibs/Documents/meter_estimation/bayesbeat/src/';
addpath(base_path);

exp = 'augmented_sampled/';

% Load model file
model_path = '/home/gigibs/Documents/meter_estimation/meter_augmentation/bayesbeat_experiments/'
model_file = strcat(model_path, exp, 'model.mat');

% load brid
split_path = '/home/gigibs/Documents/meter_estimation/meter_augmentation/data/splits/'
test_files = read_from_file(strcat(split_path, 'brid.txt'));

% where to save the results
out_folder = fullfile(pwd, exp);
disp(out_folder);

Results = test_model(model_file, test_files, out_folder);
