function lines = read_from_file(filePath)
% [lines] = read_from_file(filePath)
% 	Read content from file
% ----------------------------------------------------------------------
% INPUT Parameter:
%   filePath            : path to file
%
% OUTPUT Parameter:
%   lines               : cell array with lines content
%
% 12.08.2024 by Giovana Morais
% ----------------------------------------------------------------------
	fid = fopen(filePath, 'r');
	lines = {};

	% Check if the file was opened successfully
	if fid == -1
		error('File could not be opened: %s', filePath);
	end

	% Read the file line by line and store each line in the cell array
	tline = fgetl(fid);
	while ischar(tline)
		lines{end+1} = tline; %#ok<AGROW>
		tline = fgetl(fid);
	end

	fclose(fid);
end
