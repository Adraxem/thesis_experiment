function pareto_surface()
% PARETO_SURFACE  Interactive 3D trade-off surface for the edge-inference thesis.
%
%   Reads results/surface_data.csv (exported by export_for_matlab.py / run_all.bat)
%   and draws a parula-colored surface of SPEED vs (PEAK POWER, ENERGY/INFERENCE),
%   with the Pareto-optimal configurations marked in red.
%
%   The figure is fully interactive: drag with the mouse to rotate (rotate3d is on),
%   scroll to zoom. This is the transient-aware deployment trade-off (RQ3).
%
%   Usage:   >> pareto_surface           % from the thesis_experiment folder
%
%   Requires: the CSV above. Run run_all.bat first (or: py -3.12 export_for_matlab.py).

    here = fileparts(mfilename('fullpath'));
    csv  = fullfile(here, 'results', 'surface_data.csv');
    if ~isfile(csv)
        error(['Missing %s\n' ...
               'Run run_all.bat first, or:  py -3.12 export_for_matlab.py'], csv);
    end

    T = readtable(csv);
    x = T.p_peak_w;             % Peak power (W)
    y = T.energy_per_inf_j;     % Energy per inference (J)
    z = T.speed_score;          % Speed score (higher = faster)

    % ---- Build a smooth surface z(x, y) over a grid ------------------------
    nx = 90;  ny = 90;
    xi = linspace(min(x), max(x), nx);
    yi = linspace(min(y), max(y), ny);
    [XI, YI] = meshgrid(xi, yi);
    F  = scatteredInterpolant(x, y, z, 'natural', 'none');  % NaN outside data hull
    ZI = F(XI, YI);

    % ---- Draw --------------------------------------------------------------
    figure('Color', 'w', 'Name', 'Transient-aware deployment trade-off', ...
           'NumberTitle', 'off');
    surf(XI, YI, ZI, 'EdgeColor', 'none', 'FaceAlpha', 0.9); hold on;
    colormap(parula);            % MATLAB signature palette
    shading interp;
    cb = colorbar;  cb.Label.String = 'Speed score (surface)';

    % all configurations as faint points
    scatter3(x, y, z, 14, [0 0 0], 'filled', 'MarkerFaceAlpha', 0.25);

    % Pareto-optimal configurations
    if ismember('pareto', T.Properties.VariableNames)
        m = logical(T.pareto);
        scatter3(x(m), y(m), z(m), 80, 'r', 'filled', ...
                 'MarkerEdgeColor', 'w', 'LineWidth', 1.0, ...
                 'DisplayName', sprintf('Pareto front (n=%d)', nnz(m)));
        legend('show', 'Location', 'northeast');
    end

    xlabel('Peak power (W)');
    ylabel('Energy per inference (J)');
    zlabel('Speed score');
    title('Transient-aware deployment trade-off (parula surface)');
    view(3);  grid on;  rotate3d on;
    set(gca, 'FontSize', 11);
    hold off;
end
