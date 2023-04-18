% Robust mean estimation via Gradient Descent:
%   Y. Cheng, I. Diakonikolas, R. Ge, M. Soltanolkotabi.
%   High-Dimensional Robust Mean Estimation via Gradient Descent.
%   In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

% Input: X (N x d, N d-dimensinoal samples) and eps (fraction of corruption).
% Output: a hypothesis vector mu (a column vector).
% The number of iteration nItr is set to 10, which can be changed as you see fit.

function [mu] = robust_mean_pgd(X, eps, Itr)
    % N = number of samples, d = dimension.
    N = size(X, 1);
    d = size(X, 2);
    epsN = round(eps * N);
    
    stepSz = 1 / N;
    nItr = Itr;
    w = ones(N, 1) / N;
    for itr = 1:nItr
        % Sigma_w = X' * diag(w) * X - X' * w * w' * X;
        % [u, lambda] = eigs(Sigma_w, 1);
        Xw = X' * w;
        Sigma_w_fun = @(v) X' * (w .* (X * v)) - Xw * Xw' * v;
        [u, lambda1] = eigs(Sigma_w_fun, d, 1);

        % Compute the gradient of spectral norm (assuming unique top eigenvalue)
        % nabla_f_w = (X * u) .* (X * u) - (w' * X * u) * X * u;
        Xu = X * u;
        nabla_f_w = Xu .* Xu - 2 * (w' * Xu) * Xu;
        old_w = w;
        w = w - stepSz * nabla_f_w / norm(nabla_f_w);
        % Projecting w onto the feasible region
        w = project_onto_capped_simplex_simple(w, 1 / (N - epsN));
        
        % Use adaptive step size.
        %   If objective function decreases, take larger steps.
        %   If objective function increases, take smaller steps.
        Sigma_w_fun = @(v) X' * (w .* (X * v)) - Xw * Xw' * v;
        [~, new_lambda1] = eigs(Sigma_w_fun, d, 1);
        if (new_lambda1 < lambda1)
            stepSz = stepSz * 2;
        else
            stepSz = stepSz / 4;
            w = old_w;
        end
    end
    mu = X' * w;
end

function v = project_onto_capped_simplex_simple(w, cap)
    % The projection of w onto the capped simplex is  min(max(w - t, 0), cap)  for some scalar t
    tL = min(w) - 1;
    tR = max(w);
    for bSearch = 1:50
        t = (tL + tR) / 2;
        if (sum(min(max(w - t, 0), cap)) < 1)
            tR = t;
        else
            tL = t;
        end
    end
    v = min(max(w - t, 0), cap);
end