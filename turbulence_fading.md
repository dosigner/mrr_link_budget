FSO 난류 페이딩 및 $C_n^2$ 모델링 명세서 (Supplementary)

본 문서는 시뮬레이터의 대기 난류(Atmospheric Turbulence) 모듈 개발을 위한 상세 명세서이다.
제공된 논문 *"Modulating Retroreflector Based Free Space Optical Link for UAV-to-Ground Communications"*의 수식을 기준으로 하되, 고도 데이터 유무에 따른 $C_n^2$ 모델 스위칭 로직을 정의한다.

1. $C_n^2$ 프로파일 결정 로직 (Turbulence Profile Logic)

시뮬레이터는 입력된 파라미터(UAV 고도, 지상 풍속 등)의 가용 여부에 따라 적절한 $C_n^2$ 모델을 자동으로 선택해야 한다.

Logic Flow

Check Input: altitude_uav (UAV 고도), altitude_gs (지상국 고도), wind_speed (풍속) 값이 유효한지 확인.

Case A: 고도 정보 있음 (Vertical/Slant Path)

모델: Hufnagel-Valley (HV) 5/7 Model + Ground Layer

수식 (Ref. Eq 9 in Paper context):


$$C_n^2(h) = 0.00594(v/27)^2 (10^{-5}h)^{10} e^{-h/1000} + 2.7 \times 10^{-16} e^{-h/1500} + C_n^2(0) e^{-h/100}$$

$h$: Altitude (meters)

$v$: Wind speed (m/s)

$C_n^2(0)$: Nominal ground turbulence (typically $1.7 \times 10^{-14} m^{-2/3}$)

Case B: 고도 정보 없음 또는 수평 경로 (Horizontal/Constant Path)

모델: Constant $C_n^2$

수식: 사용자로부터 입력받은 고정값 사용 (Default: $5 \times 10^{-15} m^{-2/3}$)


$$C_n^2(h) = C_{n, constant}^2$$

2. 신틸레이션 지수 (Scintillation Index) 계산

난류 강도를 나타내는 Rytov Variance ($\sigma_R^2$)를 경로 특성에 따라 계산한다.

2.1 Uplink (Ground $\to$ UAV)

특성: 확산되는 빔이므로 구면파(Spherical Wave) 근사 적용.

Rytov Variance 식 (Ref. Eq 9):


$$\sigma_{R, up}^2 = 9 \left(\frac{2\pi}{\lambda}\right)^{7/6} (Z/Z_{h_d})^{11/6} \int_{Z_{h_g}}^{Z_{h_u}} C_n^2(h) \left(1 - \frac{h - Z_{h_g}}{Z_{h_d}}\right)^{5/6} (h - Z_{h_g})^{5/6} dh$$

$Z$: Link length (slant range)

$Z_{h_d} = Z_{h_u} - Z_{h_g}$ (고도 차이)

Constant $C_n^2$ Fallback (Case B 적용 시):

위 적분 식에서 $C_n^2(h)$를 상수로 밖으로 빼내고, 거리 $z$에 대해 단순화된 구면파 공식 사용:


$$\sigma_{R, up}^2 = 0.5 C_n^2 k^{7/6} Z^{11/6}$$

2.2 Downlink (UAV $\to$ Ground)

특성: 먼 거리에서 평행하게 내려오거나(Plane wave), MRR에 의해 반사된 빔(Pseudo-spherical)이나, 논문에서는 고고도에서 내려오는 특성을 고려하여 평면파(Plane Wave) 근사를 주로 사용 (또는 논문의 Eq 9를 Downlink 경로 적분으로 뒤집어서 적용).

Rytov Variance 식:

적분 구간을 $Z_{h_u}$에서 $Z_{h_g}$로 경로를 따라 계산 (Weighting factor $\gamma$ 변경).

Constant $C_n^2$ Fallback (Case B 적용 시):

평면파(Plane Wave) 가정 시:


$$\sigma_{R, down}^2 = 1.23 C_n^2 k^{7/6} Z^{11/6}$$

3. 확률 분포 모델링 (Fading PDF)

계산된 $\sigma_R^2$ (Rytov Variance) 값에 따라 적절한 확률 분포 모델을 적용하여 채널 계수 $h_a$를 생성한다.

3.1 Weak-to-Moderate Turbulence ($\sigma_R^2 < 1$)

모델: Log-Normal Distribution (Ref. Eq 8)

수식:


$$f_{LN}(h_a) = \frac{1}{2 h_a \sqrt{2\pi \sigma_L^2}} \exp\left( -\frac{(\ln(h_a) + 2\sigma_L^2)^2}{8\sigma_L^2} \right)$$

여기서 $\sigma_L^2 \approx \sigma_R^2 / 4$ (Log-amplitude variance)

3.2 Moderate-to-Strong Turbulence ($\sigma_R^2 \ge 1$)

모델: Gamma-Gamma Distribution (Ref. Eq 10)

수식:


$$f_{GG}(h_a) = \frac{2(\alpha\beta)^{\frac{\alpha+\beta}{2}}}{\Gamma(\alpha)\Gamma(\beta)} h_a^{\frac{\alpha+\beta}{2}-1} K_{\alpha-\beta}(2\sqrt{\alpha\beta h_a})$$

$K_v(\cdot)$: Modified Bessel function of the second kind

파라미터 $\alpha, \beta$ (Effective numbers of eddies):

Plane Wave (Downlink) 가정 시:


$$\alpha = \left[ \exp\left( \frac{0.49 \sigma_R^2}{(1 + 1.11 \sigma_R^{12/5})^{7/6}} \right) - 1 \right]^{-1}$$

$$\beta = \left[ \exp\left( \frac{0.51 \sigma_R^2}{(1 + 0.69 \sigma_R^{12/5})^{5/6}} \right) - 1 \right]^{-1}$$

(Spherical Wave Uplink의 경우 계수가 0.49/0.51 대신 0.42/0.56 등으로 변형됨을 고려하여 구현)

4. 최종 페이딩 계수 산출 (Implementation Guide)

사용자 입력 확인 (고도 유무).

calculate_cn2_profile() 호출 $\rightarrow$ 거리별 $C_n^2$ 배열 또는 상수 반환.

calculate_rytov_variance() 호출 $\rightarrow$ Uplink/Downlink 각각 $\sigma_R^2$ 산출.

$\sigma_R^2$ 크기에 따라 LogNormal 또는 GammaGamma 랜덤 변수 생성기 선택.

Uplink $h_{a_{gu}}$ 와 Downlink $h_{a_{ug}}$ 독립적으로 샘플링.