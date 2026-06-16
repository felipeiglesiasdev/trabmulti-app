import numpy as np
import matplotlib
matplotlib.use("Agg")            # backend sem janela (o Streamlit cuida de exibir)
import matplotlib.pyplot as plt
import streamlit as st
import simulacao as sim          # nossa física do Passo 2
import pandas as pd
import io
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(page_title="Escoamento Laminar 2D", page_icon="💧", layout="wide")
st.title("💧 Simulador de Escoamento Laminar 2D - Dispersão de Poluentes")
st.caption("Advecção–difusão 2D num aquário · diferenças finitas")

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
            st.markdown(
                "O escoamento é tratado como **potencial**, de modo que a pressão "
                "satisfaz a **equação de Laplace** — regime estacionário, sem fontes "
                "internas no domínio:"
            )
            st.latex(r"\nabla^2 p \;=\; \frac{\partial^2 p}{\partial x^2} "
                     r"+ \frac{\partial^2 p}{\partial y^2} \;=\; 0")

            st.markdown(
                "Como não há solução analítica para esta geometria (paredes mistas "
                "e furo de saída), resolvemos numericamente. Cada derivada segunda é "
                "aproximada por **diferenças finitas centrais de 2ª ordem** sobre a malha. "
                "Com $\\Delta x = \\Delta y$, isso leva ao **estêncil de 5 pontos**, "
                "válido em cada nó interior:"
            )
            st.latex(r"p_{i+1,j} + p_{i-1,j} + p_{i,j+1} + p_{i,j-1} - 4\,p_{i,j} = 0")
            st.markdown("ou seja, a pressão em cada ponto é a **média dos quatro vizinhos**.")

            st.markdown(
                "Aplicando essa relação a todos os "
                "$N = \\text{pontosX} \\times \\text{pontosY}$ nós, e impondo as "
                "condições de contorno em cada parede, obtemos um **sistema linear** "
                "de $N$ equações e $N$ incógnitas, escrito na forma matricial:"
            )
            st.latex(r"A\,\mathbf{p} = \mathbf{b}")
            st.markdown(
                "em que $\\mathbf{p}$ reúne as pressões desconhecidas, $A$ os "
                "coeficientes do estêncil e $\\mathbf{b}$ os termos das condições de contorno."
            )

            st.markdown(
                "A matriz $A$ é **esparsa**: cada linha tem no máximo 5 coeficientes "
                "não nulos (o nó e seus vizinhos) — todo o resto é zero. Armazená-la "
                "nesse formato é o que torna o problema tratável: na malha recomendada "
                "($480\\times160$), $A$ densa teria cerca de **5,9 bilhões** de elementos, "
                "mas apenas algumas centenas de milhares são não nulos."
            )
            st.markdown(
                "O sistema é resolvido pela função **`spsolve`** do SciPy — um **solver "
                "direto para matrizes esparsas**. Ele fatora $A$ e obtém a solução exata "
                "de $\\mathbf{p}$ em uma única chamada, **sem iteração nem critério de "
                "convergência**, ao contrário de métodos como o Gauss-Seidel."
            )
        st.markdown("**Condições de contorno**")
        st.caption("A parede direita é fixa: furo com ∂p/∂n = −J, resto ∂p/∂n = 0. "
                   "A esquerda nunca tem furo.")

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
        defaults  = [0.28, 0.37, 0.43, 0.49, 0.69]   # uma velocidade por trecho

        st.markdown("**Velocidades experimentais por trecho (cm/s)**")
        cols = st.columns(len(defaults))
        vel_exp = []
        for c, a, b, d in zip(cols, bordas_cm[:-1], bordas_cm[1:], defaults):
            vel_exp.append(
                c.number_input(f"{a}–{b} cm", min_value=0.0, value=d,
                               step=0.01, format="%.2f", key=f"vexp_{a}_{b}"))

        Z, amostras, vcal, rmse, perfil_x, perfil_v = sim.calibrar_Z(
            p, m["hx"], m["hy"], bordas_cm, vel_exp)
        ss.Z = Z   # guarda para a etapa do campo de velocidade

        c1, c2 = st.columns(2)
        c1.metric("Z ótimo", f"{Z:.4f}")
        c2.metric("Erro quadrático médio (RMSE)", f"{rmse:.4f} cm/s")

        meios = [(a + b) / 2 for a, b in zip(bordas_cm[:-1], bordas_cm[1:])]

        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
        ax.plot(meios, vel_exp, "o-", color="#4da6ff", lw=2, ms=7,
                label="Experimento")
        ax.plot(meios, vcal, "s--", color="#ffa726", lw=2, ms=7,
                label=f"Modelo calibrado (Z = {Z:.3f})")
        ax.set_xticks(bordas_cm)
        ax.set_xlabel("x (cm)", color="gray"); ax.set_ylabel("Velocidade (cm/s)", color="gray")
        ax.set_title("Calibração de Z pelas velocidades experimentais", color="white")
        ax.tick_params(colors="gray"); ax.grid(alpha=0.15, ls="--")
        ax.legend(fontsize=9, facecolor="#fff", edgecolor="#333", loc="upper left")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        fig.tight_layout(); st.pyplot(fig)


with tab5:
    st.subheader("Etapa 5 — Campo de velocidade")

    if ss.pressao is None:
        st.info("Resolva o campo de pressão na Etapa 3 primeiro.")
    elif "Z" not in ss:
        st.info("Calibre o Z na Etapa 4 primeiro.")
    else:
        st.markdown("#### 6.1 Componentes da velocidade")
        st.markdown(
            "Tratando o escoamento como **potencial**, a velocidade é o gradiente "
            "(negativo) da pressão, escalado pela condutividade $Z$ calibrada:"
        )
        st.latex(r"\vec{v} = -\,Z\,\nabla p")
        st.markdown("O gradiente de pressão é decomposto nas direções $x$ e $y$:")
        st.latex(r"v_x = -Z\,\frac{\partial p}{\partial x}, \qquad "
                 r"v_y = -Z\,\frac{\partial p}{\partial y}")
        st.markdown("Assim, basta calcular numericamente as derivadas parciais da "
                    "pressão em cada direção.")

        st.markdown("#### 6.2 Aproximação das derivadas — diferenças centrais")
        st.markdown(
            "Para obter $\\frac{\\partial p}{\\partial x}$ e "
            "$\\frac{\\partial p}{\\partial y}$ usamos **diferenças centrais de "
            "2ª ordem**, que dão boa precisão e são simétricas em relação ao ponto "
            "central. A derivada em $x$ é aproximada por:"
        )
        st.latex(r"\frac{\partial p}{\partial x}(i,j) \approx "
                 r"\frac{p_{i,j+1} - p_{i,j-1}}{2h}")
        st.markdown("e a derivada em $y$ por:")
        st.latex(r"\frac{\partial p}{\partial y}(i,j) \approx "
                 r"\frac{p_{i+1,j} - p_{i-1,j}}{2h}")
        st.markdown("Substituindo essas expressões, obtemos as fórmulas discretas "
                    "das componentes da velocidade:")
        st.latex(r"v_x(i,j) \approx -Z\,\frac{p_{i,j+1} - p_{i,j-1}}{2h}")
        st.latex(r"v_y(i,j) \approx -Z\,\frac{p_{i+1,j} - p_{i-1,j}}{2h}")
        st.markdown(
            "Essas aproximações valem nos pontos **internos** da malha. Nas **bordas**, "
            "onde não há vizinho dos dois lados, usam-se diferenças **progressivas** "
            "(para frente) ou **regressivas** (para trás). Fisicamente, o escoamento "
            "vai da região de maior para a de menor pressão, convergindo para o furo "
            "de saída."
        )

        m = ss.malha
        p = ss.pressao["p"]
        Z = ss.Z
        vx, vy = sim.campo_velocidade(p, Z, m["hx"], m["hy"])
        vel = np.sqrt(vx**2 + vy**2)
        x = np.linspace(0, m["largura"], m["nx"])
        y = np.linspace(0, m["altura"], m["ny"])

        st.metric("Z usado", f"{Z:.4f}")

        # ---- Plot 1: linhas de corrente ----
        st.markdown("##### Linhas de corrente")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        fig.patch.set_facecolor("#030303"); ax.set_facecolor("#030303")
        strm = ax.streamplot(x, y, vx, vy, color=vel, cmap="plasma",
                             density=1.2, linewidth=1.1, arrowsize=1.1)
        strm.lines.set_clim(0, np.percentile(vel, 98))
        cb = fig.colorbar(strm.lines, ax=ax, pad=0.01)
        cb.set_label("Velocidade (cm/s)", color="white"); cb.ax.tick_params(colors="white")
        ax.set_title(f"Linhas de corrente do escoamento — Z = {Z:.3f}", color="white")
        ax.set_xlabel("x (cm)", color="gray"); ax.set_ylabel("y (cm)", color="gray")
        ax.tick_params(colors="gray")
        for s in ax.spines.values(): s.set_edgecolor("#333")
        fig.tight_layout(); st.pyplot(fig)

        # amostras nas seções
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

        # ---- Plot 2: velocidade no centro ----
        with col_a:
            st.markdown("##### Velocidade no centro")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(posicoes_cm, v_centro, "s-", color="#FF6B6B", lw=2.2, ms=8,
                    mfc="white", mec="#FF6B6B", mew=2)
            ax.fill_between(posicoes_cm, v_centro, alpha=0.15, color="#FF6B6B")
            _estilo(ax, "Velocidade no centro (y = pontosY/2)")
            fig.tight_layout(); st.pyplot(fig)

        # ---- Plot 3: velocidade média em todo y ----
        with col_b:
            st.markdown("##### Velocidade média (todo y)")
            fig, ax = plt.subplots(figsize=(6, 3.2))
            fig.patch.set_facecolor("#0e1117"); ax.set_facecolor("#0e1117")
            ax.plot(posicoes_cm, v_medias, "o-", color="#00C9FF", lw=2.2, ms=8,
                    mfc="white", mec="#00C9FF", mew=2)
            ax.fill_between(posicoes_cm, v_medias, alpha=0.15, color="#00C9FF")
            _estilo(ax, "Velocidade média (todo y)")
            fig.tight_layout(); st.pyplot(fig)

        # ---- Plot 4: comparativo centro vs média ----
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

    # ===================== TERMO DIFUSIVO (só texto) =====================
    st.markdown("### 7. Termo difusivo")
    st.markdown("A difusão é o espalhamento do poluente das regiões de maior para as de "
                "menor concentração. Na equação de transporte ela aparece na parcela:")
    st.latex(r"D\,\nabla^2 u")
    st.markdown(r"onde $u(x,y,t)$ é a concentração, $D$ o coeficiente de difusão e "
                r"$\nabla^2 u$ o operador Laplaciano aplicado ao campo escalar.")

    st.markdown("#### 7.1 Discretização do Laplaciano")
    st.markdown("Em duas dimensões, o Laplaciano é:")
    st.latex(r"\nabla^2 u = \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}")
    st.markdown("Com **diferenças centrais de 2ª ordem**:")
    st.latex(r"\frac{\partial^2 u}{\partial x^2}(i,j) \approx \frac{u_{i,j+1} - 2u_{i,j} + u_{i,j-1}}{h^2}")
    st.latex(r"\frac{\partial^2 u}{\partial y^2}(i,j) \approx \frac{u_{i+1,j} - 2u_{i,j} + u_{i-1,j}}{h^2}")
    st.markdown("Somando os dois termos, chega-se ao **estêncil de 5 pontos**:")
    st.latex(r"\nabla^2 u(i,j) \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}")
    st.markdown(r"A atualização no tempo usa **Euler explícito**: "
                r"$u^{n+1}_{i,j} = u^{n}_{i,j} + \Delta t\, D\,(\nabla^2 u)^{n}_{i,j}$. "
                r"Este termo é fixo — não há nada a configurar aqui.")

    st.divider()

    # ===================== TERMO ADVECTIVO (configurável) =====================
    st.markdown("### 8. Termo advectivo")
    st.markdown("O transporte pelo escoamento aparece na parcela:")
    st.latex(r"\nabla \cdot (u\,\vec{v}) = \frac{\partial (u v_x)}{\partial x} + \frac{\partial (u v_y)}{\partial y}")
    st.markdown("A advecção **não é simétrica**: o valor transportado depende da direção "
                "da velocidade. Diferenças centrais aqui gerariam oscilações numéricas, "
                "então usamos o esquema **upwind**, que aproxima a derivada sempre pelo "
                "lado de onde o fluido vem (a montante).")

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

        fig, ax = plt.subplots(figsize=(10, 3.4))
        im = ax.imshow(u0, cmap="magma", origin="lower", aspect="equal",
                       extent=[0, m["largura"], 0, m["altura"]])
        cb = plt.colorbar(im, ax=ax, pad=0.01); cb.set_label("Concentração inicial")
        ax.set_title("Condição inicial com gotas circulares")
        ax.set_xlabel("x (cm)"); ax.set_ylabel("y (cm)")
        fig.tight_layout(); st.pyplot(fig)

        st.caption(f"{len(gotas)} gota(s).  A concentração é limitada a [0, 1].")

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
        st.markdown("**Tempo de simulação**")
        cmin, cseg = st.columns(2)
        minutos = cmin.number_input("Minutos", 0, 60, 0, 1)
        segundos = cseg.number_input("Segundos", 0, 59, 30, 1)
        tempo_total = minutos * 60 + segundos

        dt = ss.dt
        ordem = ss.ordem_upwind
        steps = int(tempo_total / dt) if dt > 0 else 0
        st.caption(f"Δt = {dt:.5f} s  ·  upwind {ordem}ª ordem  ·  "
                   f"{steps} passos para {tempo_total} s de simulação.")

        if st.button("Rodar simulação", type="primary"):
            if tempo_total <= 0:
                st.warning("Defina um tempo maior que zero.")
            else:
                vx, vy = sim.campo_velocidade(ss.pressao["p"], ss.Z, m["hx"], m["hy"])
                u = ss.u0.copy()
                max_frames = 150
                skip = max(1, steps // max_frames)
                cmap = LinearSegmentedColormap.from_list("oleo",
                    ["#000000", "#050012", "#12002b", "#240060",
                     "#35118a", "#4b2bb3", "#6a5cff"])
                u_ref = max(float(u.max()), 1e-9)

                frames = []
                prog = st.progress(0.0, "Simulando...")
                marca = max(1, steps // 100)
                for s in range(steps + 1):
                    if s % skip == 0:
                        un = np.clip(u / u_ref, 0, 1) ** 0.55
                        rgb = (cmap(un)[:, :, :3] * 255).astype("uint8")
                        frames.append(Image.fromarray(np.flipud(rgb)))
                    if s < steps:
                        u = sim.passo_tempo(u, vx, vy, ss.D, m["hx"], m["hy"], dt, ordem)
                    if s % marca == 0:
                        prog.progress(min(s / max(steps, 1), 1.0))
                prog.empty()

                buf = io.BytesIO()
                frames[0].save(buf, format="GIF", save_all=True,
                               append_images=frames[1:], duration=70, loop=0)
                ss.video_bytes = buf.getvalue()
                ss.video_info = f"{tempo_total} s · {len(frames)} frames · upwind {ordem}ª ordem"

        if ss.get("video_bytes"):
            st.markdown("##### Resultado")
            st.image(ss.video_bytes)
            st.caption(ss.get("video_info", ""))
            st.download_button("Baixar GIF", ss.video_bytes,
                               "simulacao.gif", "image/gif")