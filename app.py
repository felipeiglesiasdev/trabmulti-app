import numpy as np
import matplotlib
matplotlib.use("Agg")            # backend sem janela (o Streamlit cuida de exibir)
import matplotlib.pyplot as plt
import streamlit as st
import simulacao as sim          # nossa física do Passo 2
import pandas as pd
import io
from PIL import Image
import imageio.v2 as imageio
import tempfile, os
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Escoamento Laminar 2D", page_icon="💧", layout="wide")

st.title("💧 Simulador de Escoamento Laminar 2D - Dispersão de Poluentes")

# "Memória" da sessão: guarda resultados entre uma interação e outra
ss = st.session_state
ss.setdefault("malha", None)
ss.setdefault("pressao", None)

# Resolver a pressão é a parte cara (~2-3 s na malha cheia).
# O @st.cache_data faz o Streamlit guardar o resultado e só recalcular
# quando algum parâmetro de entrada mudar.
@st.cache_data(show_spinner=False)
def solve_pressao(nx, ny, hx, J, y1, y2, be, bs, bi):
    return sim.resolver_pressao(nx, ny, hx, J, y1, y2,bc_esq=be, bc_sup=bs, bc_inf=bi)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs(
    ["1 · Malha", "2 · Parâmetros", "3 · Campo de pressão",
     "4 · Ajuste de Z", "5 · Campo de velocidade",
     "6 · Difusão e advecção", "7 · Estabilidade", "8 · Gotas iniciais",
     "9 · Simulação"])

with tab1:
    st.subheader("Etapa 1 — Definição da malha")
    col_in, col_fig = st.columns([1, 1.4], gap="large")

    with col_in:
        st.markdown("**Valores recomendados (do experimento):**")
        st.code("larguraDominio = 60   # x (cm)\n"
                "alturaDominio  = 20   # y (cm)\n"
                "pontosX        = 480\n"
                "pontosY        = 160", language="python")

        largura = st.number_input("Largura — x (cm)", 1.0, 300.0, 60.0, 1.0)
        altura  = st.number_input("Altura — y (cm)",  1.0, 300.0, 20.0, 1.0)
        nx = st.number_input("Pontos em x (pontosX)", 10, 1000, 480, 10)
        ny = st.number_input("Pontos em y (pontosY)", 10, 1000, 160, 10)

        hx, hy = largura / nx, altura / ny
        c1, c2, c3 = st.columns(3)
        c1.metric("Δx (cm)", f"{hx:.4f}")
        c2.metric("Δy (cm)", f"{hy:.4f}")
        c3.metric("Nós", f"{nx*ny}")

        # guarda a malha na memória da sessão
        ss.malha = dict(largura=largura, altura=altura,
                        nx=int(nx), ny=int(ny), hx=hx, hy=hy)

    with col_fig:
        fig, ax = plt.subplots(figsize=(7, 3))
        for xc in np.linspace(0, largura, min(int(nx), 25)):
            ax.axvline(xc, color="#86efac", lw=0.6)
        for yc in np.linspace(0, altura, min(int(ny), 17)):
            ax.axhline(yc, color="#86efac", lw=0.6)
        ax.add_patch(plt.Rectangle((0, 0), largura, altura,
                                   fill=False, edgecolor="#16a34a", lw=2))
        ax.set_xlim(-1, largura + 1); ax.set_ylim(-1, altura + 1)
        ax.set_aspect("equal")
        ax.set_title(f"Malha {int(nx)} × {int(ny)}")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        fig.tight_layout()
        st.pyplot(fig)
        st.caption("Grade ilustrativa (linhas reduzidas pra visualização). "
                   "Δx e Δy ao lado são os valores reais.")
        

with tab2:
    st.subheader("Etapa 2 — Parâmetros físicos")
    st.write("Ambos obtidos experimentalmente.")

    c1, c2 = st.columns(2)
    with c1:
        D = st.number_input("D — coef. de difusão (cm²/s)",
                            min_value=0.0, value=0.0034, step=0.0001, format="%.4f")
    with c2:
        J = st.number_input("J — intensidade de fluxo no furo",
                            min_value=0.0, value=50.0, step=1.0)

    ss.D, ss.J = float(D), float(J)
    st.info(f"Em uso:  D = {D:.4f} cm²/s   ·   J = {J:g}")


with tab3:
    st.subheader("Etapa 3 — Campo de pressão  (∇²p = 0)")

    if ss.malha is None:
        st.info("Defina a malha na Etapa 1 primeiro.")
    else:
        m = ss.malha
        J = ss.get("J", 50.0)
        opc = {"∂p/∂n = 0 (impermeável)": "neumann", "p = 0 (Dirichlet)": "zero"}
        with st.expander("Como o campo de pressão é resolvido", expanded=True):

            st.markdown(r"O campo de pressão satisfaz a **equação de Laplace**:")
            st.latex(r"\nabla^2 p = \frac{\partial^2 p}{\partial x^2} + "
                    r"\frac{\partial^2 p}{\partial y^2} = 0")
            st.markdown(r"As derivadas segundas são aproximadas por **diferenças "
                        r"centrais de 2ª ordem** em cada direção:")
            st.latex(r"\frac{\partial^2 p}{\partial x^2}(i,j) \approx "
                    r"\frac{p_{i,j+1} - 2p_{i,j} + p_{i,j-1}}{h^2}, \qquad "
                    r"\frac{\partial^2 p}{\partial y^2}(i,j) \approx "
                    r"\frac{p_{i+1,j} - 2p_{i,j} + p_{i-1,j}}{h^2}")
            st.markdown(r"Substituindo na equação de Laplace e isolando $p_{i,j}$, "
                        r"cada nó interno resulta na **média dos quatro vizinhos**:")
            st.latex(r"p_{i,j} = \frac{p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1}}{4}")
        cc1, cc2, cc3 = st.columns(3)
        be = opc[cc1.selectbox("Parede esquerda", list(opc), index=1)]  # default p=0
        bs = opc[cc2.selectbox("Parede superior", list(opc), index=0)]  # default neumann
        bi = opc[cc3.selectbox("Parede inferior", list(opc), index=0)]

        st.markdown("**Método de resolução**")
        metodo = st.radio(
            "Solver do sistema linear",
            ["spsolve (direto, esparso)",
             "Gauss-Seidel (iterativo)",
             "Método 3",
             "Método 4"],
            captions=["Disponível",
                      "Novos métodos serão implementados no futuro",
                      "Novos métodos serão implementados no futuro",
                      "Novos métodos serão implementados no futuro"],
            index=0,
        )
        if metodo != "spsolve (direto, esparso)":
            st.warning("Esse método ainda não está disponível. "
                       "Novos métodos serão implementados no futuro. "
                       "Usando **spsolve** por enquanto.")

        raio = st.slider("Raio do furo (cm)", 0.5, 5.0, 1.25, 0.25)

        if st.button("Resolver pressão", type="primary"):
            y1, y2 = sim.indices_furo(m["altura"], m["hy"], raio)
            with st.spinner("Resolvendo ∇²p = 0 ..."):
                p = solve_pressao(m["nx"], m["ny"], m["hx"], J, y1, y2, be, bs, bi)
            ss.pressao = dict(p=p, J=J)

        if ss.pressao is not None:
            p = ss.pressao["p"]
            ext = [0, m["largura"], 0, m["altura"]]
            fig, ax = plt.subplots(figsize=(9, 3.2))
            fig.patch.set_facecolor("#171717"); ax.set_facecolor("#171717")
            im = ax.imshow(p, cmap="jet", origin="lower", aspect="auto", extent=ext)
            cb = fig.colorbar(im, ax=ax, pad=0.01)
            cb.set_label("Pressão", color="white"); cb.ax.tick_params(colors="white")
            ax.set_title(f"Campo de pressão — J = {ss.pressao['J']:g}", color="white")
            ax.set_xlabel("x (cm)", color="gray"); ax.set_ylabel("y (cm)", color="gray")
            ax.tick_params(colors="gray")
            fig.tight_layout()
            st.pyplot(fig)
            st.success(f"p mínima: {p.min():.3f}   ·   p máxima: {p.max():.3f} "
                       f"  (média ancorada em 0)")
            

with tab4:
    st.subheader("Etapa 4 — Ajuste de Z (calibração)")

    if ss.pressao is None:
        st.info("Resolva o campo de pressão na Etapa 3 primeiro.")
    else:
        m = ss.malha
        p = ss.pressao["p"]
        bordas_cm = [30, 35, 40, 45, 50, 55]
        defaults  = [0.28, 0.37, 0.43, 0.49, 0.69]

        st.markdown("Informe a velocidade **medida no experimento** em cada trecho de "
                    "5 cm. O app ajusta automaticamente o **Z** que faz o modelo chegar "
                    "mais perto dessas medidas.")

        st.markdown("**Velocidades experimentais por trecho (cm/s)**")
        cols = st.columns(len(defaults))
        vel_exp = []
        for c, a, b, d in zip(cols, bordas_cm[:-1], bordas_cm[1:], defaults):
            vel_exp.append(
                c.number_input(f"{a}–{b} cm", min_value=0.0, value=d,
                               step=0.01, format="%.2f", key=f"vexp_{a}_{b}"))
        st.caption("Cada campo é a velocidade média do escoamento naquele intervalo de x.")

        Z, amostras, vcal, rmse, perfil_x, perfil_v = sim.calibrar_Z(
            p, m["hx"], m["hy"], bordas_cm, vel_exp)
        ss.Z = Z

        st.markdown("")  # respiro
        col_z, col_e = st.columns(2)
        with col_z:
            st.markdown(
                f"""
                <div style="background:#0e1117;border:1px solid #16a34a;
                            border-left:6px solid #16a34a;border-radius:10px;padding:14px 18px;">
                    <div style="color:#86efac;font-size:.72rem;text-transform:uppercase;
                                letter-spacing:.12em;font-weight:600;">Z calibrado (ótimo)</div>
                    <div style="color:#fff;font-size:2rem;font-weight:700;line-height:1.1;">{Z:.4f}</div>
                    <div style="color:#9aa6b2;font-size:.75rem;margin-top:4px;">
                        valor que melhor reproduz as velocidades medidas</div>
                </div>
                """, unsafe_allow_html=True)
        with col_e:
            st.markdown(
                f"""
                <div style="background:#0e1117;border:1px solid #333;
                            border-left:6px solid #ffa726;border-radius:10px;padding:14px 18px;">
                    <div style="color:#ffd9a6;font-size:.72rem;text-transform:uppercase;
                                letter-spacing:.12em;font-weight:600;">Erro médio (RMSE)</div>
                    <div style="color:#fff;font-size:2rem;font-weight:700;line-height:1.1;">
                        {rmse:.4f} <span style="font-size:1rem;color:#9aa6b2;">cm/s</span></div>
                    <div style="color:#9aa6b2;font-size:.75rem;margin-top:4px;">
                        distância média modelo × experimento · menor = melhor</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("")  # respiro
        meios = [(a + b) / 2 for a, b in zip(bordas_cm[:-1], bordas_cm[1:])]
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.plot(meios, vel_exp, "o-", color="#4da6ff", lw=2, ms=7, label="Experimento")
        ax.plot(meios, vcal, "s--", color="#ffa726", lw=2, ms=7,
                label=f"Modelo calibrado (Z = {Z:.3f})")
        ax.set_xticks(bordas_cm)
        ax.set_xlabel("x (cm)", color="gray"); ax.set_ylabel("Velocidade (cm/s)", color="gray")
        ax.set_title("Calibração de Z pelas velocidades experimentais", color="white")
        ax.tick_params(colors="gray"); ax.grid(alpha=0.15, ls="--")
        ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333", labelcolor="white", loc="upper left")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        fig.tight_layout(); st.pyplot(fig)

with tab5:
    st.subheader("Etapa 5 — Campo de velocidade")

    if ss.pressao is None:
        st.info("Resolva o campo de pressão na Etapa 3 primeiro.")
    elif "Z" not in ss:
        st.info("Calibre o Z na Etapa 4 primeiro.")
    else:
        m = ss.malha
        p = ss.pressao["p"]
        Z = ss.Z

        # ---- Z em destaque ----
        st.markdown(
            f"""
            <div style="background:#0e1117;border:1px solid #16a34a;
                        border-left:6px solid #16a34a;border-radius:10px;
                        padding:14px 18px;margin-bottom:16px;">
                <div style="color:#86efac;font-size:.72rem;text-transform:uppercase;
                            letter-spacing:.12em;font-weight:600;">
                    Condutividade calibrada (Etapa 4)</div>
                <div style="color:#fff;font-size:2rem;font-weight:700;line-height:1.1;">
                    Z = {Z:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- Formulação (compacta) ----
        st.markdown("**Componentes da velocidade**")
        st.markdown("Tratando o escoamento como **potencial**, a velocidade é o "
                    "gradiente (negativo) da pressão, escalado pela condutividade "
                    "$Z$. O gradiente é decomposto em $x$ e $y$:")
        st.latex(r"\vec{v} = -\,Z\,\nabla p \qquad\Longrightarrow\qquad "
                 r"v_x = -Z\,\frac{\partial p}{\partial x}, \quad "
                 r"v_y = -Z\,\frac{\partial p}{\partial y}")

        st.markdown("**Aproximação das derivadas — diferenças centrais de 2ª ordem**")
        st.markdown("As derivadas parciais são aproximadas por diferenças centrais "
                    "(boa precisão e simétricas em torno do ponto):")
        st.latex(r"\frac{\partial p}{\partial x}(i,j) \approx "
                 r"\frac{p_{i,j+1} - p_{i,j-1}}{2h} \qquad "
                 r"\frac{\partial p}{\partial y}(i,j) \approx "
                 r"\frac{p_{i+1,j} - p_{i-1,j}}{2h}")
        st.markdown("Substituindo, chega-se às fórmulas discretas das componentes:")
        st.latex(r"v_x(i,j) \approx -Z\,\frac{p_{i,j+1} - p_{i,j-1}}{2h} \qquad "
                 r"v_y(i,j) \approx -Z\,\frac{p_{i+1,j} - p_{i-1,j}}{2h}")
        st.caption("Nos pontos internos usam-se diferenças centrais; nas bordas, "
                   "diferenças progressivas ou regressivas. O escoamento vai da maior "
                   "para a menor pressão, convergindo para o furo de saída.")

        st.divider()

        vx, vy = sim.campo_velocidade(p, Z, m["hx"], m["hy"])
        vel = np.sqrt(vx**2 + vy**2)
        x = np.arange(m["nx"]) * m["hx"]
        y = np.arange(m["ny"]) * m["hy"]

        # ---- Plot 1: linhas de corrente ----
        st.markdown("##### Linhas de corrente")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        fig.patch.set_facecolor("#030303"); ax.set_facecolor("#030303")
        strm = ax.streamplot(x, y, vx, vy, color=vel, cmap="plasma",
                             density=1.2, linewidth=0.9, arrowsize=1.1)
        strm.lines.set_clim(0, np.percentile(vel, 98))
        cb = fig.colorbar(strm.lines, ax=ax, pad=0.01)
        cb.set_label("Velocidade (cm/s)", color="white"); cb.ax.tick_params(colors="white")
        ax.set_title(f"Linhas de corrente do escoamento — Z = {Z:.3f}", color="white")
        ax.set_xlabel("x (cm)", color="gray"); ax.set_ylabel("y (cm)", color="gray")
        ax.tick_params(colors="gray")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        fig.tight_layout(); st.pyplot(fig)

        posicoes_cm = [30, 35, 40, 45, 50, 55]
        idx = [min(int(xx / m["hx"]), m["nx"] - 1) for xx in posicoes_cm]
        v_medias = [float(vel[:, i].mean()) for i in idx]
        v_centro = [float(vel[m["ny"] // 2, i]) for i in idx]

        def _estilo(ax, titulo):
            ax.set_title(titulo, color="white")
            ax.set_xlabel("x (cm)", color="gray"); ax.set_ylabel("Velocidade (cm/s)", color="gray")
            ax.tick_params(colors="gray"); ax.grid(alpha=0.15, ls="--")
            for s in ax.spines.values(): s.set_edgecolor("#333")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("##### Velocidade no centro")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(posicoes_cm, v_centro, "s-", color="#FF6B6B", lw=2.2, ms=8,
                    mfc="white", mec="#FF6B6B", mew=2)
            ax.fill_between(posicoes_cm, v_centro, alpha=0.15, color="#FF6B6B")
            _estilo(ax, "Velocidade no centro (y = pontosY/2)")
            fig.tight_layout(); st.pyplot(fig)
        with col_b:
            st.markdown("##### Velocidade média (todo y)")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(posicoes_cm, v_medias, "o-", color="#00C9FF", lw=2.2, ms=8,
                    mfc="white", mec="#00C9FF", mew=2)
            ax.fill_between(posicoes_cm, v_medias, alpha=0.15, color="#00C9FF")
            _estilo(ax, "Velocidade média (todo y)")
            fig.tight_layout(); st.pyplot(fig)

        st.markdown("##### Centro vs Média")
        fig, ax = plt.subplots(figsize=(9, 3.4))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.plot(posicoes_cm, v_centro, "s-", color="#FF6B6B", lw=2.2, ms=8,
                mfc="white", mec="#FF6B6B", mew=2, label="Centro (y = pontosY/2)")
        ax.plot(posicoes_cm, v_medias, "o-", color="#00C9FF", lw=2.2, ms=8,
                mfc="white", mec="#00C9FF", mew=2, label="Média (todo y)")
        _estilo(ax, "Velocidade no centro vs média em y")
        ax.legend(fontsize=9, facecolor="#1a1a1a", edgecolor="#333")
        fig.tight_layout(); st.pyplot(fig)


with tab6:
    st.subheader("Etapa 6 — Termos difusivo e advectivo")

    # ===================== TERMO DIFUSIVO (acordeão) =====================
    with st.expander("Termo difusivo", expanded=False):
        st.markdown("A difusão é o espalhamento do poluente das regiões de maior para "
                    "as de menor concentração. Na equação de transporte ela aparece na "
                    "parcela:")
        st.latex(r"D\,\nabla^2 u")
        st.markdown(r"onde $u(x,y,t)$ é a concentração, $D$ o coeficiente de difusão e "
                    r"$\nabla^2 u$ o operador Laplaciano aplicado ao campo escalar.")
        st.markdown("**Discretização do Laplaciano**")
        st.markdown("Em duas dimensões, o Laplaciano é:")
        st.latex(r"\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}")
        st.markdown("Com **diferenças centrais de 2ª ordem**:")
        st.latex(r"\frac{\partial^2 u}{\partial x^2}(i,j) \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2}")
        st.latex(r"\frac{\partial^2 u}{\partial y^2}(i,j) \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}")
        st.markdown("Somando os dois termos, chega-se à média ponderada dos vizinhos:")
        st.latex(r"\nabla^2 u(i,j) \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}")
        st.markdown(r"A atualização no tempo usa **Euler explícito**: "
                    r"$u^{n+1}_{i,j} = u^{n}_{i,j} + \Delta t\, D\,(\nabla^2 u)^{n}_{i,j}$. "
                    r"Este termo é fixo — não há nada a configurar aqui.")

    # ===================== TERMO ADVECTIVO (acordeão) =====================
    with st.expander("Termo advectivo", expanded=True):
        st.markdown("O transporte pelo escoamento aparece na parcela:")
        st.latex(r"\nabla \cdot (u\,\vec{v}) = \frac{\partial (u v_x)}{\partial x} + \frac{\partial (u v_y)}{\partial y}")
        st.markdown("A advecção **não é simétrica**: o valor transportado depende da "
                    "direção da velocidade. Diferenças centrais aqui gerariam oscilações "
                    "numéricas, então usamos o esquema **upwind**, que aproxima a derivada "
                    "sempre pelo lado de onde o fluido vem (a montante).")

        st.markdown("**Escolha o esquema upwind:**")
        ordem_label = st.radio(
            "Ordem do upwind",
            ["Upwind de 1ª ordem", "Upwind de 2ª ordem"],
            index=1, horizontal=True, label_visibility="collapsed",
        )
        ss.ordem_upwind = 1 if "1ª" in ordem_label else 2

        if ss.ordem_upwind == 1:
            st.markdown(r"Derivada em $x$ (1ª ordem) — regressiva se $v_x>0$, progressiva se $v_x<0$:")
            st.latex(r"\frac{\partial u}{\partial x} \approx "
                     r"\begin{cases} \dfrac{u_{i,j} - u_{i,j-1}}{h}, & v_x > 0 \\[6pt] "
                     r"\dfrac{u_{i,j+1} - u_{i,j}}{h}, & v_x < 0 \end{cases}")
            st.markdown(r"Derivada em $y$ (1ª ordem) — regressiva se $v_y>0$, progressiva se $v_y<0$:")
            st.latex(r"\frac{\partial u}{\partial y} \approx "
                     r"\begin{cases} \dfrac{u_{i,j} - u_{i-1,j}}{h}, & v_y > 0 \\[6pt] "
                     r"\dfrac{u_{i+1,j} - u_{i,j}}{h}, & v_y < 0 \end{cases}")
            st.caption("Mais robusto e estável, porém mais difusivo (espalha um pouco a mais).")
        else:
            st.markdown(r"Derivada em $x$ (2ª ordem) — usa dois pontos a montante:")
            st.latex(r"\frac{\partial u}{\partial x} \approx "
                     r"\begin{cases} \dfrac{3u_{i,j} - 4u_{i,j-1} + u_{i,j-2}}{2h}, & v_x > 0 \\[6pt] "
                     r"\dfrac{-3u_{i,j} + 4u_{i,j+1} - u_{i,j+2}}{2h}, & v_x < 0 \end{cases}")
            st.markdown(r"Derivada em $y$ (2ª ordem):")
            st.latex(r"\frac{\partial u}{\partial y} \approx "
                     r"\begin{cases} \dfrac{3u_{i,j} - 4u_{i-1,j} + u_{i-2,j}}{2h}, & v_y > 0 \\[6pt] "
                     r"\dfrac{-3u_{i,j} + 4u_{i+1,j} - u_{i+2,j}}{2h}, & v_y < 0 \end{cases}")
            st.caption("Mais preciso (menos difusão numérica), mantém melhor a forma da mancha.")

        st.markdown("O termo advectivo entra no avanço temporal como:")
        st.latex(r"-\left( v_x\,\frac{\partial u}{\partial x} + v_y\,\frac{\partial u}{\partial y} \right)")
        st.info(f"Esquema selecionado: **upwind de {ss.ordem_upwind}ª ordem** "
                f"(será usado no passo de tempo da próxima etapa).")
    


with tab7:
    st.subheader("Etapa 7 — Estabilidade (definição do Δt)")

    if ss.pressao is None:
        st.info("Resolva o campo de pressão na Etapa 3 primeiro.")
    elif "Z" not in ss:
        st.info("Calibre o Z na Etapa 4 primeiro.")
    else:
        st.markdown("Como o avanço no tempo é **Euler explícito**, o passo Δt precisa "
                    "respeitar **dois** limites de estabilidade ao mesmo tempo:")
        st.markdown("**Difusão:**")
        st.latex(r"\Delta t \le \frac{h^2}{4D}")
        st.markdown("**Advecção (condição CFL):**")
        st.latex(r"\Delta t \le \frac{1}{\dfrac{|v_x|_{max}}{\Delta x} + \dfrac{|v_y|_{max}}{\Delta y}}")
        st.markdown("O Δt permitido é o **menor** dos dois.")

        m = ss.malha
        D = ss.get("D", 0.0034)
        vx, vy = sim.campo_velocidade(ss.pressao["p"], ss.Z, m["hx"], m["hy"])
        vx_max = float(np.abs(vx).max())
        vy_max = float(np.abs(vy).max())

        dt_dif = m["hx"]**2 / (4 * D)
        dt_adv = 1.0 / ((vx_max / m["hx"]) + (vy_max / m["hy"]))
        dt_max = min(dt_dif, dt_adv)
        gargalo = "advecção (CFL)" if dt_adv < dt_dif else "difusão"

        c1, c2, c3 = st.columns(3)
        c1.metric("Δt máx · difusão", f"{dt_dif:.4f} s")
        c2.metric("Δt máx · advecção", f"{dt_adv:.5f} s")
        c3.metric("Δt máximo permitido", f"{dt_max:.5f} s")
        st.caption(f"Limitante: **{gargalo}**  ·  "
                   f"|vx|max = {vx_max:.3f} cm/s, |vy|max = {vy_max:.3f} cm/s")

        st.markdown("**Fator de segurança** — fração do Δt máximo a usar de fato:")
        fator = st.slider("Fator de segurança", 0.05, 0.95, 0.40, 0.05)
        dt = fator * dt_max
        ss.dt = dt

        st.success(f"Δt escolhido = {fator:.2f} × {dt_max:.5f} = **{dt:.5f} s**  "
                   f"→ estável (abaixo do limite).")
        st.caption("Quanto menor o fator, mais estável e mais lenta a simulação "
                   "(mais passos para o mesmo tempo físico).")
        

with tab8:
    st.subheader("Etapa 8 — Condição inicial (gotas)")

    if ss.malha is None:
        st.info("Defina a malha na Etapa 1 primeiro.")
    else:
        m = ss.malha
        st.markdown("Cada gota é um círculo de concentração. Edite a tabela — use o "
                    "**+** para adicionar gotas e selecione linhas para remover.")

        if "gotas_df" not in ss:
            ss.gotas_df = pd.DataFrame(
                [{"x (cm)": 30.0, "y (cm)": 10.0, "raio (cm)": 2.0, "concentração": 1.0}])

        df = st.data_editor(
            ss.gotas_df, num_rows="dynamic", key="editor_gotas",
            column_config={
                "x (cm)": st.column_config.NumberColumn(min_value=0.0, max_value=float(m["largura"]), step=1.0),
                "y (cm)": st.column_config.NumberColumn(min_value=0.0, max_value=float(m["altura"]), step=1.0),
                "raio (cm)": st.column_config.NumberColumn(min_value=0.1, step=0.25),
                "concentração": st.column_config.NumberColumn(min_value=0.0, max_value=1.0, step=0.1),
            },
        )

        df = df.dropna()
        gotas = [{"x": float(r["x (cm)"]), "y": float(r["y (cm)"]),
                  "raio": float(r["raio (cm)"]), "conc": float(r["concentração"])}
                 for _, r in df.iterrows()]

        x = np.linspace(0, m["largura"], m["nx"])
        y = np.linspace(0, m["altura"], m["ny"])
        X_cm, Y_cm = np.meshgrid(x, y, indexing="xy")
        u0 = sim.condicao_inicial(m["nx"], m["ny"], X_cm, Y_cm, gotas)
        ss.u0 = u0   # guarda para o passo de tempo

        oil = LinearSegmentedColormap.from_list("oleo",
                ["#e8d49a", "#cda34a", "#9c6b2e", "#5e3a17"]).with_extremes(bad="#c6e4f5")
        disp = np.where(u0 > 1e-9, u0, np.nan)   # onde não há óleo -> água
        fig, ax = plt.subplots(figsize=(10, 3.4))
        ax.set_facecolor("#c6e4f5")
        im = ax.imshow(disp, cmap=oil, origin="lower", aspect="equal",
                       extent=[0, m["largura"], 0, m["altura"]], vmin=0, vmax=1)
        cb = plt.colorbar(im, ax=ax, pad=0.01); cb.set_label("Concentração")
        ax.set_title("Gotas (concentração)")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        fig.tight_layout(); st.pyplot(fig)


with tab9:
    st.subheader("Etapa 9 — Simulação e vídeo")

    faltam = []
    if ss.pressao is None: faltam.append("pressão (Etapa 3)")
    if "Z" not in ss: faltam.append("Z (Etapa 4)")
    if "dt" not in ss: faltam.append("Δt (Etapa 7)")
    if "u0" not in ss: faltam.append("gotas (Etapa 8)")
    if "ordem_upwind" not in ss: faltam.append("upwind (Etapa 6)")

    if faltam:
        st.info("Antes de simular, conclua: " + ", ".join(faltam) + ".")
    else:
        m = ss.malha
        dt = ss.dt
        ordem = ss.ordem_upwind

        st.markdown("**Tempo de simulação**")
        cmin, cseg = st.columns(2)
        minutos = cmin.number_input("Minutos", 0, 60, 0, 1)
        segundos = cseg.number_input("Segundos", 0, 59, 30, 1)
        tempo_total = minutos * 60 + segundos

        limiar = st.slider("Limiar de visualização", 0.0, 0.5, 0.05, 0.01,
                           help="Concentração abaixo deste valor aparece como água (fundo azul).")

        steps = int(tempo_total / dt) if dt > 0 else 0
        st.caption(f"Δt = {dt:.5f} s  ·  upwind {ordem}ª ordem  ·  "
                   f"{steps} passos para {tempo_total} s de simulação.")

        if st.button("Rodar simulação", type="primary"):
            if tempo_total <= 0:
                st.warning("Defina um tempo maior que zero.")
            else:
                vx, vy = sim.campo_velocidade(ss.pressao["p"], ss.Z, m["hx"], m["hy"])
                u = ss.u0.copy()

                fps_alvo = 25
                skip = max(1, round(1 / (fps_alvo * dt)))
                fps_real = 1 / (skip * dt)

                oil = LinearSegmentedColormap.from_list("oleo_video",
                    ["#e8d49a", "#cda34a", "#9c6b2e", "#5e3a17"])
                bg_rgb = np.array([198, 228, 245], dtype="uint8")   # água
                u_ref = max(float(u.max()), 1e-9)

                # faixas de medição -> colunas + cor das linhas
                faixas_cm = [30, 35, 40, 45, 50, 55]
                cols_faixa = [min(int(round(xc / m["hx"])), m["nx"] - 1) for xc in faixas_cm]
                vermelho = np.array([220, 30, 30], dtype="uint8")

                def _render(u):
                    un = np.clip(u / u_ref, 0, 1) ** 0.6
                    rgb = (oil(un)[:, :, :3] * 255).astype("uint8")
                    rgb[u < limiar] = bg_rgb
                    return np.flipud(rgb)

                def _add_faixas(frame):
                    f = frame.copy()
                    tracos = (np.arange(f.shape[0]) // 6) % 2 == 0   # tracejado vertical
                    for c in cols_faixa:
                        f[tracos, c] = vermelho
                        if c + 1 < f.shape[1]:
                            f[tracos, c + 1] = vermelho              # 2px de espessura
                    return f

                frames1, frames2 = [], []
                prog = st.progress(0.0, "Simulando...")
                marca = max(1, steps // 100)
                for s in range(steps + 1):
                    if s % skip == 0:
                        fr = _render(u)
                        frames1.append(fr)
                        frames2.append(_add_faixas(fr))
                    if s < steps:
                        u = sim.passo_tempo(u, vx, vy, ss.D, m["hx"], m["hy"], dt, ordem)
                    if s % marca == 0:
                        prog.progress(min(s / max(steps, 1), 1.0))
                prog.empty()

                with st.spinner("Codificando os vídeos (MP4)..."):
                    def _mp4(frames):
                        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
                        tmp.close()
                        imageio.mimsave(tmp.name, frames, format="FFMPEG",
                                        fps=fps_real, codec="libx264", quality=8)
                        with open(tmp.name, "rb") as f:
                            data = f.read()
                        os.remove(tmp.name)
                        return data
                    ss.video_bytes = _mp4(frames1)
                    ss.video_bytes2 = _mp4(frames2)
                ss.video_info = (f"{tempo_total} s de simulação · {len(frames1)} frames · "
                                 f"{fps_real:.1f} fps · upwind {ordem}ª ordem")

        if ss.get("video_bytes"):
            st.markdown("##### Vídeo 1 — dispersão")
            st.video(ss.video_bytes)
            st.caption(ss.get("video_info", ""))
            st.download_button("Baixar MP4 (vídeo 1)", ss.video_bytes,
                               "simulacao.mp4", "video/mp4")

        if ss.get("video_bytes2"):
            st.markdown("##### Vídeo 2 — com as faixas de medição")
            st.video(ss.video_bytes2)
            st.caption("Linhas pontilhadas vermelhas em x = 30, 35, 40, 45, 50 e 55 cm.")
            st.download_button("Baixar MP4 (vídeo 2)", ss.video_bytes2,
                               "simulacao_faixas.mp4", "video/mp4")