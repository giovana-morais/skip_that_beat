function Results = train_and_test_hmm(test_files, train_files, out_folder)
% [Results] = train_and_test_hmm(in_file, out_folder)
%   Train an HMM and test it.
% ----------------------------------------------------------------------
% INPUT Parameter:
%   in_file             : audio file
%   out_folder          : folder, where output (beats and model) are stored
%   train_files         : cell array with training audio files
%
% OUTPUT Parameter:
%   Results             : structure array with beat tracking results
%
% 12.08.2024 by Giovana Morais
% ----------------------------------------------------------------------

	% Train dataset
	% Path to lab file
	Params.trainLab = train_files;
	Params.testLab = test_files;
	Params.data_path = out_folder;
	Params.results_path = out_folder;
	Params.save_beats = 1;
	Params.save_downbeats = 1;
	Params.save_rhythm = 1;
	Params.save_meter = 1;

	% TRAINING THE MODEL
	% create beat tracker object
	BT = BeatTracker(Params);
	% train model
	BT.train_model();

	% TEST THE MODEL
	% do beat tracking
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

% Load file with the training data
split_path = '/home/gigibs/Documents/meter_estimation/meter_augmentation/splits/'
exp = 'augmented_sampled';

train_files = read_from_file(strcat(split_path, exp, '_train.txt'));
val_files = read_from_file(strcat(split_path, exp, '_val.txt'));
train_files = [train_files, val_files];
test_files = read_from_file(strcat(split_path, exp, '_test.txt'));

% where to save the results
out_folder = fullfile(pwd, exp);

Results = train_and_test_hmm(test_files, train_files, out_folder);
