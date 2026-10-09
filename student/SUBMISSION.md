# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Hoàng Đức Dũng
- MSSV: 2A202602798
- Email: 26ai.dunghd2@vinuni.edu.vn
- Link repo (fork): https://github.com/hddung-vinai/K4-L2L3-DAY23-HoangDucDung-2A202602798-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): commit `CP6` cuối cùng trên `main` (hash 40 ký tự nộp trên LMS); artifacts sinh ở commit CP5 `c2c18b30710d0b5747b002515edc2164b7a632bb`

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, `[0, 198]`, `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, `0`
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, 519 / 16 / 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.3436, 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.2668, 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss: Fused có RMSE thấp hơn LiDAR 0.0145 m (0.1359 so với 0.1503, khoảng −10%), trong khi `matches` (502), `ghost_track_frames` (0), `missed_gt_frames` (239) và `mean_confirmed_tracks` (2.52) giống hệt nhau ở hai mode. Vì vậy RMSE hai mode được tính trên cùng tập cặp ghép nên so sánh công bằng: chênh lệch chỉ đến từ độ chính xác vị trí, không phải từ việc một mode giữ ít track khó hơn. Điều này đúng thiết kế track-then-fuse: camera chỉ update EKF, không cộng/trừ score, không tạo/xoá track, nên vòng đời track hai mode trùng nhau. Theo từng frame, fused có `sum_sq_err` nhỏ hơn lidar ở 137/195 frame có ghép, nhưng không phải lúc nào cũng tốt hơn: frame 4 lidar 0.0113 so với fused 0.0320, và lần chạy thử 20 frame đầu fused còn kém hơn (0.154 so với 0.114 m), nên không kết luận từ một đoạn ngắn. Precision tracking = 502/(502+0) = 1.00 cho cả hai mode: 16 FP của detector không bao giờ đủ hit liên tiếp để thành confirmed. Coverage = 502/519 = 0.97. Trong 239 lần miss, 217 nằm ở frame detector bỏ sót xe (`det_fn > 0`, ví dụ frame 40–42 có `det_fn = 1`, `misses = 1`); 22 lần còn lại là độ trễ xác nhận track mới (frame 0–3 đầu segment, 44–47 khi có xe mới xuất hiện) vì track cần 4 hit LiDAR mới vượt ngưỡng 0.8. Tức là phần lớn miss do giới hạn recall của detector (0.70), không phải do tracker làm mất track.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - **LiDAR**: `z = (x, y, z)` tâm hộp 3D (mét, hệ LiDAR), `R = diag(0.1², 0.1², 0.1²)`. `h(x)` tuyến tính (chỉ xoay/tịnh tiến vị trí) nên `H` là ma trận hằng 3×6 `[R_rot | 0]`.
   - **Camera**: `z = (u, v)` pixel, `R = diag(5², 5²)` từ `sigma_cam_i/j` (`build_camera_measurement`). `h(x)` là chiếu pinhole `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (`camera_measurement_prediction`), phi tuyến vì chia cho độ sâu `x_s`, nên `H` là Jacobian 2×6 tính tại `x` (platform `Sensor.get_H`). Camera không đo độ sâu: nó chỉ ràng buộc hướng nhìn, nên chủ yếu sửa vị trí ngang/dọc chứ không sửa khoảng cách.
   - Cùng một hàm `ekf_update` (`kalman.py`) dùng cho cả hai, chỉ khác `z`, `R`, `h(x)`, `H` do `meas.sensor` cung cấp.

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Tránh gán đo vào track sai: đo của xe khác hoặc FP ở xa sẽ kéo EKF lệch và có thể làm hỏng cả track. Cặp có `d² = γᵀS⁻¹γ` vượt `chi2.ppf(0.995, dim_meas)` (≈12.84 cho lidar 3D, ≈10.60 cho camera 2D) bị đặt cost `inf` (`association_cost_matrix`, `chi2_gate`).
   - Dùng Mahalanobis thay vì Euclid vì nó chuẩn hoá theo `S = HPHᵀ + R`: track mới (P lớn) có cổng rộng, track ổn định có cổng hẹp; và so được cả đo mét (lidar) lẫn pixel (camera). Nếu cặp đúng thì `d²` theo phân phối χ² với số bậc tự do = số chiều đo, nên cổng 99.5% chỉ loại nhầm 0.5% cặp đúng.
   - Trước Mahalanobis còn kiểm tra FOV (`meas.sensor.in_fov(track.x)`): track ở sau camera có độ sâu ≤ 0, chiếu pinhole không xác định.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - **Track-then-fuse**: chỉ có một danh sách track; mỗi frame predict một lần, rồi `associate_and_update` với LiDAR, rồi `associate_and_update` với camera trên cùng các track (`run_lab.py` gọi lần lượt với `lidar_sensor` và `camera_sensor`). Không gộp raw point cloud với ảnh trước detection.
   - Trên log: ở mọi frame, `confirmed`, `matches`, `ghosts`, `misses` của record `"mode": "fused"` bằng đúng record `"mode": "lidar"` (tổng 502 matches, 0 ghost, 239 miss ở cả hai), chỉ `sum_sq_err` khác (ví dụ frame 4: lidar 0.0113, fused 0.0320). Nghĩa là camera không đổi tập track (do LiDAR quyết định) mà chỉ tinh chỉnh vị trí của track đã có.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Innovation camera `γ = z − h(x)` sẽ có bias (trung bình khác 0, lệch có hệ thống theo `u` hoặc `v`) thay vì nhiễu quanh 0. Lệch xoay cho bias gần như đều về một phía; lệch tịnh tiến cho bias lớn hơn ở xe gần (vì chia cho độ sâu).
   - `d²` Mahalanobis của camera thường xuyên lớn hơn kỳ vọng χ²(2) (trung bình > 2), nhiều cặp bị cổng loại → ít camera update; các cặp lọt cổng kéo vị trí track lệch nên RMSE fused tăng và có thể vượt RMSE lidar. Residual LiDAR ở frame kế tiếp cũng lớn lên vì hai cảm biến kéo track về hai hướng.
   - Cách phát hiện: theo dõi trung bình/histogram `γ` camera, tỷ lệ cặp bị gate, và so RMSE fused với lidar (ở đây fused tốt hơn 0.0145 m nên không thấy dấu hiệu lệch đáng kể).

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Khi `meas_list` rỗng thì không lấy được sensor từ `meas_list[0].sensor`, nhưng vẫn phải gọi `manager.manage_tracks(unassigned_tracks, [], sensor)`: lượt LiDAR thì mọi track trong FOV bị trừ `1/window` (miss) và có thể bị xoá; lượt camera thì không được đụng vào vòng đời. Không có sensor tường minh thì frame LiDAR không có detection bị bỏ qua và track ma sống mãi, hoặc lượt camera rỗng bị xử lý nhầm thành miss LiDAR.
   - LiDAR quyết định score/init/delete vì nó đo vị trí 3D đầy đủ, phủ quanh xe và là nguồn detection thật (FPN-ResNet). Camera chỉ đo 2D, không có độ sâu nên không đủ để khởi tạo track 3D; chỉ phủ camera FRONT nên nếu cho trừ score thì track ngoài tầm camera bị phạt sai; và trong lab đo camera lấy từ GT + nhiễu, nếu cho cộng score sẽ nuôi track bằng thông tin GT. Vì vậy `handle_updated_track` và `manage_tracks` của platform `return` khi `sensor.name != "lidar"`; camera chỉ đi qua `filter_obj.update`.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Khởi tạo** (`init_track_state_from_meas`): đo LiDAR chưa ghép tạo track `initialized`, `score = 1/6`, vị trí = `z` đổi sang hệ xe, vận tốc 0 với `P` vận tốc `diag(50², 50², 5²)`.
   - **Xác nhận** (`update_track_score`): mỗi hit LiDAR `score += 1/6` (tối đa 1.0), track chưa confirmed thành `tentative`; khi `score > 0.8` thì `confirmed`. Từ 1/6 cần 4 hit (5/6 ≈ 0.83) → trên log track confirmed từ frame 4, nên frame 0–3 có `misses = 2`.
   - **Giữ confirmed sau miss**: miss trong FOV LiDAR thì `score −= 1/6`, nhưng track đã `confirmed` không bị hạ trạng thái (1.0 → 0.83 vẫn confirmed). Track ngoài FOV LiDAR không bị trừ.
   - **Xoá** (`should_delete_track`, các điều kiện OR): `P[0,0]` hoặc `P[1,1] > max_P = 9` (σ vị trí > 3 m) bất kể score; confirmed có `score < 0.6`; chưa confirmed có `score ≤ 0`. Camera không bao giờ gọi các hàm này.


## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Claude Code trong VS Code)
- Dùng cho phần nào (hàm, câu hỏi, debug): giải thích đề; viết code Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`); debug môi trường (pytest crash do OpenMP trên Windows → `KMP_DUPLICATE_LIB_OK=TRUE`; `float()` trên ma trận 1×1 với numpy mới → `.item()`); sắp xếp dữ liệu vào `data/`; đối chiếu metrics và soạn nháp phần tóm tắt kết quả + 6 câu giải thích. Lần chạy Waymo do tôi tự chạy; mọi số liệu lấy từ `metrics.json`/`grade_run.log` của chính lần chạy đó.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): chạy `pytest student/tests` sau mỗi Part (test E–H đều pass, không còn xfailed); tính tay test số EKF (ví dụ `P'[0,0] = 1 + 0.2²·4 + 0.6 = 1.76`); chạy `fusion-run-lab --fusion compare --seed 0` thử 20 frame rồi cả segment 0–198; tính lại RMSE, matches, ghost, miss từ `grade_run.log` và khớp `metrics.json`; đọc lại từng hàm, đối chiếu công thức trong `docs/HUONG_DAN_KY_THUAT.md` và gợi ý `# vi: TODO`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
