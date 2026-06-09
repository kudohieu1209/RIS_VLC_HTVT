# RIS_VLC_Simulation

Project Python mô phỏng hệ thống truyền thông ánh sáng nhìn thấy trong nhà
(Visible Light Communication - VLC) có hỗ trợ RIS. Mục tiêu là so sánh kênh
VLC LoS thông thường, trường hợp LoS bị vật cản che khuất, và trường hợp RIS
tạo đường phản xạ NLoS AP-RIS-PD để cải thiện SNR và data rate.

## Mục Tiêu Mô Phỏng

- Mô phỏng hệ thống VLC trong phòng 5 m x 5 m x 3 m.
- Tính channel gain, received optical power, SNR và data rate.
- So sánh 4 kịch bản:
  - Không có vật cản, không RIS.
  - Có vật cản, không RIS.
  - Có vật cản, có RIS.
  - Không có vật cản, có RIS.
- Quét vị trí RIS trên tường `y = 0` để tìm vị trí tối ưu theo data rate.
- Xuất bảng CSV, báo cáo Markdown và hình ảnh biểu đồ để dùng trong báo cáo.

## Cấu Trúc Project

```text
RIS_VLC_Simulation/
|-- main.py                      # wrapper tương thích cách chạy cũ
|-- app.py                       # dashboard Streamlit + Plotly
|-- pyproject.toml               # metadata package và console script
|-- requirements.txt
|-- requirements-dev.txt
|-- example_config.json
|-- ris_vlc_sim/
|   |-- cli.py
|   |-- config.py
|   |-- vlc_channel.py
|   |-- ris_channel.py
|   |-- simulation.py
|   |-- plotting.py
|   |-- report.py
|   `-- utils.py
|-- tests/
|-- figures/
`-- results/
```

Các file `config.py`, `simulation.py`, `plotting.py`, `utils.py`,
`vlc_channel.py`, `ris_channel.py` ở thư mục gốc hiện là wrapper tương thích.
Code chính nằm trong package `ris_vlc_sim/`.

## Cài Đặt

Cài thư viện runtime:

```bash
pip install -r requirements.txt
```

Nếu muốn dùng lệnh console `ris-vlc-sim` và chạy test:

```bash
pip install -e .
pip install -r requirements-dev.txt
```

## Cách Chạy

Cách chạy cũ vẫn dùng được:

```bash
python main.py
```

Chạy dashboard tương tác:

```bash
streamlit run app.py
```

Chạy qua module/package:

```bash
python -m ris_vlc_sim
```

Sau khi cài editable package, có thể chạy:

```bash
ris-vlc-sim
```

Mở thư mục hình sau khi chạy xong:

```bash
python main.py --open-figures
```

Chỉ sinh CSV và báo cáo, bỏ qua hình để chạy nhanh:

```bash
python main.py --no-plots
```

Ghi output sang thư mục khác:

```bash
python main.py --output-dir outputs/demo
```

Dùng file cấu hình JSON để override tham số:

```bash
python main.py --config example_config.json --output-dir outputs/example
```

Dashboard cho phép chỉnh vị trí PD, thông số RIS, công suất LED, FoV, bandwidth,
noise variance và độ phân giải quét. Kết quả hiển thị bằng bảng, KPI, bar chart,
heatmap tối ưu RIS, SNR map và mô hình phòng 3D có thể xoay bằng chuột.

File JSON dùng dạng flat key đúng tên trong `SimulationConfig`:

```json
{
  "pd_position": [2.5, 1.0, 0.85],
  "ris_x_points": 31,
  "ris_z_points": 31,
  "pd_grid_points": 40,
  "led_transmit_power_w": 1.0,
  "noise_variance": 1e-14
}
```

Nếu key không tồn tại, vector 3D sai kích thước, góc không hợp lệ, số điểm
quét nhỏ hơn 2 hoặc `noise_variance <= 0`, chương trình sẽ báo lỗi rõ ràng.

## Kiểm Thử

Chạy toàn bộ test:

```bash
pytest
```

Các test hiện kiểm tra:

- Công thức Lambertian và optical concentrator.
- Giao tuyến đoạn thẳng với vật cản dạng hộp 3D.
- Blockage mặc định của đường AP-PD.
- Metric khi channel gain bằng 0.
- 4 scenario chính và hiệu quả RIS khi LoS bị chặn.
- Optimization RIS nằm trong biên quét.
- CLI smoke test với grid nhỏ và `--no-plots`.

## Công Thức Sử Dụng

### Bậc Lambertian

```text
m = -ln(2) / ln(cos(Phi_half))
```

Trong đó `Phi_half` là góc bán công suất của LED.

### Kênh LoS

```text
H_LoS = ((m + 1) * A_PD / (2*pi*d^2)) * cos(phi)^m * Ts * g(psi) * cos(psi)
```

Trong đó:

- `d` là khoảng cách AP-PD.
- `phi` là góc phát xạ so với trục LED.
- `psi` là góc tới tại PD.
- Nếu `psi > FoV` thì `H_LoS = 0`.
- LED hướng xuống theo trục `-z`.
- PD hướng lên theo trục `+z`.

### Bộ Tập Trung Quang

```text
g(psi) = n^2 / sin(FoV)^2, nếu 0 <= psi <= FoV
g(psi) = 0, nếu psi > FoV
```

### Mô Hình Vật Cản

Vật cản được mô hình hóa bằng một khối hộp chữ nhật 3D:

```text
obstacle_min = (2.15, 1.65, 0.85) m
obstacle_max = (2.85, 1.95, 2.45) m
```

Với mỗi vị trí PD, chương trình kiểm tra đoạn thẳng AP-PD có cắt khối vật cản
hay không. Nếu có cắt thì LoS bị chặn hoàn toàn.

```text
B = 1: LoS không bị che khuất
B = 0: LoS bị chặn hoàn toàn
H_LoS_eff = B * H_LoS
```

### Kênh RIS

RIS được đặt trên tường `y = 0` và hướng vào phòng theo vector pháp tuyến `+y`.
Mô hình phản xạ đơn giản:

```text
H_RIS = rho_RIS * H_AP_RIS * H_RIS_PD * G_align
```

Trong project này:

- `rho_RIS = 0.8`.
- `G_align = 1` khi RIS được giả định căn chỉnh tốt về phía PD.
- `H_AP_RIS` là độ lợi hình học từ AP đến RIS.
- `H_RIS_PD` là độ lợi hình học từ RIS đến PD.
- `ris_effective_area = 2.0 m^2` là diện tích hiệu dụng xấp xỉ của RIS.

Lưu ý quan trọng: đây là mô hình xấp xỉ đơn giản phục vụ bài tập lớn, không
phải mô hình phần cứng RIS quang học đầy đủ.

### Tổng Kênh, SNR Và Data Rate

```text
H_total_without_RIS = H_LoS_eff
H_total_with_RIS = H_LoS_eff + H_RIS
Pr = Pt * H_total
SNR = (Rp * Pr)^2 / noise_variance
SNR_dB = 10 * log10(SNR)
Rb = Bmod * log2(1 + SNR)
```

Nếu `SNR = 0`, chương trình đặt `SNR_dB = -inf` trong bảng kết quả. Khi vẽ
heatmap, giá trị `-inf` được thay bằng mức sàn `-60 dB`.

## File Kết Quả

### `results/scenario_results.csv`

Bảng so sánh 4 kịch bản, gồm các cột:

- `scenario`
- `H_LoS`
- `H_RIS`
- `H_total`
- `Pr_W`
- `SNR_linear`
- `SNR_dB`
- `data_rate_bps`
- `data_rate_Mbps`
- `LoS_blocked`

### `results/ris_position_optimization.csv`

Bảng kết quả quét vị trí RIS trên tường `y = 0`, gồm `x_RIS_m`, `z_RIS_m`,
`H_RIS`, `SNR_dB` và `data_rate_Mbps`.

### `results/simulation_summary.md`

Báo cáo tóm tắt tự sinh, gồm thông số hình học, vị trí RIS tối ưu, bảng kết quả
4 kịch bản và mức cải thiện data rate khi có RIS trong trường hợp có vật cản.

### `figures/`

Thư mục hình gồm:

- `snr_comparison.png`
- `data_rate_comparison.png`
- `ris_position_optimization_3d.png`
- `ris_position_optimization_heatmap.png`
- `system_geometry_3d.png`
- `room_top_view.png`
- `snr_heatmap_without_ris.png`
- `snr_heatmap_with_ris.png`

## Giả Thiết Và Giới Hạn Mô Hình

- Phòng có kích thước 5 m x 5 m x 3 m.
- AP LED đặt tại `(2.5, 2.5, 3.0) m`.
- PD mặc định đặt tại `(2.5, 1.0, 0.85) m`.
- Mặt phẳng người dùng có `z = 0.85 m`.
- LED hướng xuống theo trục `-z`; PD hướng lên theo trục `+z`.
- RIS nằm trên tường `y = 0`.
- RIS được mô hình hóa xấp xỉ như đường phản xạ AP-RIS-PD.
- Vật cản được mô hình hóa bằng khối hộp 3D; hệ số blockage `B` được xác định
  bằng kiểm tra giao tuyến giữa đoạn AP-PD và khối vật cản.
- Mô hình dùng để phục vụ bài tập lớn và minh họa xu hướng cải thiện nhờ RIS,
  chưa phải mô hình phần cứng RIS quang học đầy đủ.
