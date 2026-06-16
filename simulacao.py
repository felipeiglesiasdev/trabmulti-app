import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def indices_furo(alturaDominio, passoY, raio_furo):
    y_centro = alturaDominio / 2
    j_centro = round(y_centro / passoY)
    j_raio   = round(raio_furo / passoY)
    return j_centro - j_raio, j_centro + j_raio

def resolver_pressao(pontosX, pontosY, passoX, J, y1, y2, bc_esq='zero', bc_sup='neumann', bc_inf='neumann'):
    N = pontosX * pontosY
    # Converte a coordenada (i, j) da grade num índice único de 0 a N-1
    def idx(i, j):
        return j * pontosX + i
    A = lil_matrix((N, N))   # matriz do sistema
    b = np.zeros(N)          # lado direito
    for j in range(pontosY):
        for i in range(pontosX):
            k = idx(i, j)
            is_esq = (i == 0)
            is_dir = (i == pontosX - 1)
            is_inf = (j == 0)            and not is_esq and not is_dir
            is_sup = (j == pontosY - 1)  and not is_esq and not is_dir
            is_furo = is_dir and (y1 <= j < y2)
            is_interior = not (is_esq or is_dir or is_inf or is_sup)

            # ---- parede esquerda (configurável, sem furo) ----
            if is_esq:
                if bc_esq == 'zero':
                    A[k, k] = 1.0
                else:  # neumann
                    A[k, idx(0, j)] =  1.0
                    A[k, idx(1, j)] = -1.0
                b[k] = 0.0

            # ---- furo na parede direita:  ∂p/∂n = -J ----
            elif is_furo:
                A[k, idx(pontosX - 1, j)] =  1.0
                A[k, idx(pontosX - 2, j)] = -1.0
                b[k] = passoX * (-J)

            # ---- resto da parede direita:  ∂p/∂n = 0 ----
            elif is_dir:
                A[k, idx(pontosX - 1, j)] =  1.0
                A[k, idx(pontosX - 2, j)] = -1.0
                b[k] = 0.0

            # ---- parede inferior (configurável) ----
            elif is_inf:
                if bc_inf == 'zero':
                    A[k, k] = 1.0
                else:
                    A[k, idx(i, 0)] =  1.0
                    A[k, idx(i, 1)] = -1.0
                b[k] = 0.0

            # ---- parede superior (configurável) ----
            elif is_sup:
                if bc_sup == 'zero':
                    A[k, k] = 1.0
                else:
                    A[k, idx(i, pontosY - 1)] =  1.0
                    A[k, idx(i, pontosY - 2)] = -1.0
                b[k] = 0.0

            # ---- interior: Laplaciano 5 pontos ----
            elif is_interior:
                A[k, idx(i + 1, j)] =  1.0
                A[k, idx(i - 1, j)] =  1.0
                A[k, idx(i, j + 1)] =  1.0
                A[k, idx(i, j - 1)] =  1.0
                A[k, k]             = -4.0
    p_vec = spsolve(A.tocsr(), b)        # resolve o sistema
    p = p_vec.reshape((pontosY, pontosX))
    p -= np.mean(p)                       # ancora a média em zero
    return p


def campo_velocidade(p, Z, passoX, passoY):
    vx = np.zeros_like(p)
    vy = np.zeros_like(p)
    vx[:, 1:-1] = -Z * (p[:, 2:] - p[:, :-2]) / (2 * passoX)
    vy[1:-1, :] = -Z * (p[2:, :] - p[:-2, :]) / (2 * passoY)
    vx[:, 0]  = -Z * (p[:, 1]  - p[:, 0])  / passoX
    vx[:, -1] = -Z * (p[:, -1] - p[:, -2]) / passoX
    vy[0, :]  = -Z * (p[1, :]  - p[0, :])  / passoY
    vy[-1, :] = -Z * (p[-1, :] - p[-2, :]) / passoY
    return vx, vy

def calibrar_Z(p, passoX, passoY, bordas_cm, vel_exp):
    pontosY, pontosX = p.shape
    vx1, vy1 = campo_velocidade(p, 1.0, passoX, passoY)   # Z = 1
    vel_base = np.sqrt(vx1**2 + vy1**2)
    y_central = pontosY // 2
    linha = vel_base[y_central, :]
    amostras_base = []
    for a, b in zip(bordas_cm[:-1], bordas_cm[1:]):
        ia = min(int(round(a / passoX)), pontosX - 1)
        ib = min(int(round(b / passoX)), pontosX - 1)
        if ib <= ia:
            ib = ia + 1
        amostras_base.append(linha[ia:ib].mean())
    amostras_base = np.array(amostras_base)
    vel_exp = np.asarray(vel_exp, dtype=float)
    Z_otimo = float(np.sum(amostras_base * vel_exp) / np.sum(amostras_base**2))
    vel_calibrada = Z_otimo * amostras_base
    rmse = float(np.sqrt(np.mean((vel_calibrada - vel_exp)**2)))
    perfil_x = np.arange(pontosX) * passoX
    perfil_v = Z_otimo * linha
    return Z_otimo, amostras_base, vel_calibrada, rmse, perfil_x, perfil_v


def aplicar_contorno_u(u):
    u[:, 0]  = u[:, 1]
    u[:, -1] = u[:, -2]
    u[0, :]  = u[1, :]
    u[-1, :] = u[-2, :]
    return u


def condicao_inicial(pontosX, pontosY, X_cm, Y_cm, gotas):
    u = np.zeros((pontosY, pontosX))
    for g in gotas:
        mascara = (X_cm - g['x'])**2 + (Y_cm - g['y'])**2 <= g['raio']**2
        u[mascara] = g['conc']
    u = np.clip(u, 0, 1)
    return aplicar_contorno_u(u)


def passo_difusao(u, D, h, dt):
    u_new = u.copy()
    lap = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2]
           - 4 * u[1:-1, 1:-1]) / (h * h)
    u_new[1:-1, 1:-1] += dt * D * lap
    return aplicar_contorno_u(u_new)


def passo_adveccao(u, vx, vy, h, dt, ordem=2):
    u_new = u.copy()
    dudx = np.zeros_like(u)
    dudy = np.zeros_like(u)
    if ordem == 1:
        dudx[:, 1:-1] = np.where(vx[:, 1:-1] > 0,
                                 (u[:, 1:-1] - u[:, 0:-2]) / h,
                                 (u[:, 2:] - u[:, 1:-1]) / h)
        dudy[1:-1, :] = np.where(vy[1:-1, :] > 0,
                                 (u[1:-1, :] - u[0:-2, :]) / h,
                                 (u[2:, :] - u[1:-1, :]) / h)
    else:
        dudx[:, 2:-2] = np.where(vx[:, 2:-2] > 0,
                                 (3*u[:, 2:-2] - 4*u[:, 1:-3] + u[:, 0:-4]) / (2*h),
                                 (-3*u[:, 2:-2] + 4*u[:, 3:-1] - u[:, 4:]) / (2*h))
        dudy[2:-2, :] = np.where(vy[2:-2, :] > 0,
                                 (3*u[2:-2, :] - 4*u[1:-3, :] + u[0:-4, :]) / (2*h),
                                 (-3*u[2:-2, :] + 4*u[3:-1, :] - u[4:, :]) / (2*h))

    u_new -= dt * (vx * dudx + vy * dudy)
    return u_new


def passo_tempo(u, vx, vy, D, passoX, passoY, dt, ordem=2):
    u = passo_difusao(u, D, passoX, dt)
    u = passo_adveccao(u, vx, vy, passoX, dt, ordem)
    return u