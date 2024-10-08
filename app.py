import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time


def mixer_simulation(initial_masses, flow_rates, mixing_time, num_steps):
    components = ["C", "Fe", "Mn", "Cr"]
    num_components = len(components)

    masses = np.zeros((num_steps, num_components))
    masses[0] = initial_masses

    total_mass = np.sum(initial_masses)
    dt = mixing_time / num_steps

    for i in range(1, num_steps):
        for j in range(num_components):
            inflow = flow_rates[j] * dt
            outflow = (masses[i - 1, j] / total_mass) * np.sum(flow_rates) * dt
            masses[i, j] = masses[i - 1, j] + inflow - outflow

        total_mass = np.sum(masses[i])

    return masses


def create_mixer_diagram():
    fig, ax = plt.subplots(figsize=(8, 6))

    # 믹서 본체
    mixer = plt.Rectangle((0.3, 0.1), 0.4, 0.6, fill=False)
    ax.add_patch(mixer)

    # 입구 파이프
    ax.arrow(
        0.1, 0.8, 0.2, 0, width=0.02, head_width=0.05, head_length=0.05, fc="b", ec="b"
    )
    ax.arrow(
        0.1, 0.6, 0.2, 0, width=0.02, head_width=0.05, head_length=0.05, fc="r", ec="r"
    )
    ax.arrow(
        0.1, 0.4, 0.2, 0, width=0.02, head_width=0.05, head_length=0.05, fc="g", ec="g"
    )
    ax.arrow(
        0.1, 0.2, 0.2, 0, width=0.02, head_width=0.05, head_length=0.05, fc="y", ec="y"
    )

    # 출구 파이프
    ax.arrow(
        0.7,
        0.4,
        0.2,
        0,
        width=0.02,
        head_width=0.05,
        head_length=0.05,
        fc="purple",
        ec="purple",
    )

    # 레이블
    ax.text(0.05, 0.8, "C", fontsize=12, va="center")
    ax.text(0.05, 0.6, "Fe", fontsize=12, va="center")
    ax.text(0.05, 0.4, "Mn", fontsize=12, va="center")
    ax.text(0.05, 0.2, "Cr", fontsize=12, va="center")
    ax.text(0.95, 0.4, "Mixture", fontsize=12, va="center")

    ax.text(0.5, 0.9, "Mixer", fontsize=16, ha="center")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    return fig


st.title("화학공학 실험: 연속 혼합 반응기(CSTR) 시뮬레이션")

st.write(
    """
안녕하세요, 여러분. 오늘 우리는 화학공학에서 매우 중요한 장치 중 하나인 연속 혼합 반응기(Continuous Stirred Tank Reactor, CSTR)의 동적 거동을 시뮬레이션해 볼 것입니다.

이 실험의 목적은 다음과 같습니다:
1. CSTR 내부의 물질 수지 이해
2. 시간에 따른 반응기 내 조성 변화 관찰
3. 정상 상태(Steady State) 도달 과정 분석

우리의 CSTR은 4개의 성분(C, Fe, Mn, Cr)을 다루고 있으며, 각 성분은 서로 다른 유입 속도로 반응기에 주입됩니다. 이 과정은 다음의 물질 수지 방정식으로 설명됩니다:
"""
)

st.latex(r"\frac{dm_i}{dt} = F_{i,in} - \frac{m_i}{M_{total}} \sum_{j} F_{j,out}")

st.write(
    """
여기서:
- $m_i$: 반응기 내 성분 $i$의 질량
- $F_{i,in}$: 성분 $i$의 유입 속도
- $M_{total}$: 반응기 내 총 질량
- $\sum_{j} F_{j,out}$: 총 유출 속도

이 방정식의 물리적 의미는 다음과 같습니다:

1. 반응기로 들어오는 물질의 양 ($F_{i,in}$)
2. 반응기에서 나가는 물질의 양 ($\\frac{m_i}{M_{total}} \\sum_{j} F_{j,out}$)
3. 이 두 항의 차이가 반응기 내 물질의 축적 속도 ($\\frac{dm_i}{dt}$)

이 시뮬레이션을 통해 우리는 초기 조건과 유입 속도가 시스템의 동적 거동과 최종 평형 상태에 어떤 영향을 미치는지 관찰할 수 있습니다. 
이는 실제 공정에서 반응기의 설계와 운전 조건 최적화에 매우 중요한 정보를 제공합니다.

자, 이제 시뮬레이션을 시작해 봅시다!
"""
)

st.sidebar.header("입력 매개변수")
initial_masses = [
    st.sidebar.number_input(f"{comp} 초기 질량 (kg)", value=val, step=0.1)
    for comp, val in zip(["C", "Fe", "Mn", "Cr"], [5.0, 15.0, 10.0, 20.0])
]
flow_rates = [
    st.sidebar.number_input(f"{comp} 유입 속도 (kg/s)", value=val, step=0.01)
    for comp, val in zip(["C", "Fe", "Mn", "Cr"], [0.2, 0.1, 0.15, 0.05])
]
mixing_time = st.sidebar.number_input("혼합 시간 (s)", value=50.0, step=1.0)
num_steps = 50

st.write("### CSTR 시뮬레이션 개략도")
diagram = create_mixer_diagram()
st.pyplot(diagram)

if st.button("시뮬레이션 시작"):
    masses = mixer_simulation(initial_masses, flow_rates, mixing_time, num_steps)
    components = ["C", "Fe", "Mn", "Cr"]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 18))

    progress_bar = st.progress(0)
    status_text = st.empty()
    plot_spot = st.empty()

    for i in range(num_steps):
        # 막대 그래프 업데이트
        ax1.clear()
        ax1.bar(components, masses[i])
        ax1.set_ylim(0, np.max(masses) * 1.1)
        ax1.set_title(
            f"Component Masses in Reactor (Time: {i*mixing_time/num_steps:.2f} s)"
        )
        ax1.set_ylabel("Mass (kg)")
        ax1.set_xlabel("Components")

        # 파이 차트 업데이트
        ax2.clear()
        ax2.pie(masses[i], labels=components, autopct="%1.1f%%", startangle=90)
        ax2.set_title(
            f"Mass Fractions in Reactor (Time: {i*mixing_time/num_steps:.2f} s)"
        )

        # 선 그래프 업데이트
        ax3.clear()
        for j, comp in enumerate(components):
            ax3.plot(
                np.linspace(0, i * mixing_time / num_steps, i + 1),
                masses[: i + 1, j],
                label=comp,
            )
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Mass (kg)")
        ax3.set_title("Component Masses Over Time")
        ax3.legend()

        plot_spot.pyplot(fig)
        progress_bar.progress((i + 1) / num_steps)
        status_text.text(f"{int((i+1)/num_steps*100)}% 완료")

        time.sleep(0.05)

    st.write(f"유입 속도: {flow_rates} kg/s")
    st.write(f"총 시뮬레이션 시간: {mixing_time} s")

    st.write(
        """
    ### 시뮬레이션 결과 분석
    
    우리가 방금 관찰한 시뮬레이션 결과를 분석해 봅시다:

    1. 초기 과도 상태:
       - 시뮬레이션 초기에 각 성분의 질량이 급격하게 변하는 것을 볼 수 있습니다.
       - 이는 초기 조건과 유입 속도의 차이로 인한 것으로, 시스템이 평형 상태로 이동하는 과정입니다.

    2. 정상 상태 도달:
       - 시간이 지남에 따라 각 성분의 질량 변화율이 감소하고, 결국 일정한 값으로 수렴합니다.
       - 이 상태가 바로 정상 상태(Steady State)로, 유입되는 물질의 양과 유출되는 물질의 양이 평형을 이루는 상태입니다.

    3. 최종 조성:
       - 정상 상태에서의 각 성분의 질량 분율은 유입 속도에 비례합니다.
       - 유입 속도가 빠른 성분일수록 최종 조성에서 더 큰 비율을 차지하게 됩니다.

    4. 시스템의 응답 시간:
       - 그래프의 기울기를 통해 시스템의 응답 속도를 파악할 수 있습니다.
       - 초기에 급격한 변화를 보이다가 점차 완만해지는 것을 관찰할 수 있습니다.

    이러한 관찰 결과는 실제 화학 공정 설계와 운영에 중요한 통찰을 제공합니다:
    
    - 반응기의 크기와 유입 속도를 조절하여 원하는 조성의 제품을 얻을 수 있습니다.
    - 공정의 시작과 정지, 또는 운전 조건 변경 시 시스템이 안정화되는 데 필요한 시간을 예측할 수 있습니다.
    - 다양한 운전 조건에서의 시스템 거동을 예측하여 최적의 운전 조건을 결정할 수 있습니다.

    이 시뮬레이션을 통해 여러분은 연속 혼합 반응기의 기본 원리와 동적 거동을 이해했을 것입니다. 
    이러한 이해는 향후 더 복잡한 반응 시스템과 공정을 다루는 데 기초가 될 것입니다.
    """
    )

st.write(
    """
화학공정에서 CSTR은 매우 중요한 장치입니다. 이 시뮬레이션에서 배운 원리는 다음과 같은 실제 응용 분야에 적용됩니다:

1. 화학 반응기 설계 및 최적화
2. 폐수 처리 시스템
3. 연속 발효 공정
4. 중합 반응기 설계
5. 제약 산업에서의 연속 생산 공정

앞으로의 학습에서 이러한 응용 분야에 대해 더 자세히 다루게 될 것입니다. 
오늘의 실험이 여러분의 이해에 도움이 되었기를 바랍니다. 질문이 있다면 언제든 물어보세요!
"""
)
