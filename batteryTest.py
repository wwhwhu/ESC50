import subprocess
import time

# 连接ADB
adb_connect_command = "adb connect 192.168.1.50"
subprocess.run(adb_connect_command, shell=True)

# 等待连接成功
time.sleep(3)  # 可根据实际情况调整等待时间，确保连接成功

# 重置电量统计
adb_reset_command = "adb shell dumpsys batterystats --reset"
subprocess.run(adb_reset_command, shell=True)

# 等待5分钟
time.sleep(300)  # 等待5分钟

# 查找电量统计信息
adb_stats_command = "adb shell dumpsys batterystats | findstr u0a239"
subprocess.run(adb_stats_command, shell=True)

# 断开ADB连接
adb_disconnect_command = "adb disconnect"
subprocess.run(adb_disconnect_command, shell=True)

# 8.10 CNN_original
# UID u0a239: 6.90 fg: 5.17 ( cpu=5.44 (5m 11s 540ms) cpu:fg=5.17 (1m 14s 873ms) audio=1.46 (4m 53s 118ms) system_services=0.00477 )
# UID u0a239: 9.99 fg: 5.00 ( cpu=8.53 (8m 26s 357ms) cpu:fg=5.00 (1m 13s 697ms) audio=1.46 (4m 53s 540ms) system_services=0.00712 )
# UID u0a239: 7.42 fg: 5.00 ( cpu=5.95 (5m 40s 970ms) cpu:fg=5.00 (1m 12s 196ms) audio=1.46 (4m 53s 538ms) system_services=0.00569 )

# 7.25 CNN_quant1
# UID u0a239: 7.14 fg: 5.41 ( cpu=5.68 (4m 53s 851ms) cpu:fg=5.41 (1m 10s 622ms) audio=1.46 (4m 53s 519ms) system_services=0.00511 )
# UID u0a239: 6.89 fg: 5.11 ( cpu=5.42 (5m 12s 18ms) cpu:fg=5.11 (1m 14s 823ms) audio=1.46 (4m 53s 656ms) system_services=0.00508 )
# UID u0a239: 7.73 fg: 6.02 ( cpu=6.26 (4m 20s 969ms) cpu:fg=6.02 (1m 1s 589ms) audio=1.46 (4m 54s 100ms) system_services=0.00501 )

# 7.13 CNN_quant2.0
# UID u0a239: 7.11 fg: 5.39 ( cpu=5.65 (4m 55s 953ms) cpu:fg=5.39 (1m 10s 858ms) audio=1.46 (4m 53s 461ms) system_services=0.00394 )
# UID u0a239: 7.38 fg: 5.77 ( cpu=5.91 (4m 28s 504ms) cpu:fg=5.77 (1m 5s 280ms) audio=1.46 (4m 54s 33ms) system_services=0.00497 )
# UID u0a239: 6.89 fg: 5.00 ( cpu=5.42 (5m 19s 577ms) cpu:fg=5.00 (1m 14s 432ms) audio=1.46 (4m 53s 395ms) system_services=0.00435 )

# 6.96 CNN_quant3
# UID u0a239: 6.93 fg: 5.14 ( cpu=5.46 (5m 6s 223ms) cpu:fg=5.14 (1m 13s 10ms) audio=1.46 (4m 53s 302ms) system_services=0.00406 )
# UID u0a239: 6.94 fg: 5.24 ( cpu=5.48 (4m 55s 551ms) cpu:fg=5.24 (1m 12s 443ms) audio=1.46 (4m 53s 628ms) system_services=0.00455 )
# UID u0a239: 7.02 fg: 5.02 ( cpu=5.56 (5m 17s 802ms) cpu:fg=5.02 (1m 10s 902ms) audio=1.46 (4m 53s 365ms) system_services=0.00460 )

# 7.21 CNN_prune1_ex
# UID u0a239: 7.07 fg: 5.32 ( cpu=5.61 (5m 1s 423ms) cpu:fg=5.32 (1m 12s 509ms) audio=1.46 (4m 53s 578ms) system_services=0.00438 )
# UID u0a239: 7.12 fg: 5.05 ( cpu=5.66 (5m 31s 413ms) cpu:fg=5.05 (1m 13s 942ms) audio=1.46 (4m 53s 330ms) system_services=0.00482 )
# UID u0a239: 7.44 fg: 4.97 ( cpu=5.98 (5m 47s 401ms) cpu:fg=4.97 (1m 11s 922ms) audio=1.46 (4m 53s 645ms) system_services=0.00430 )

# 7.41 CNN_prune2_ex
# UID u0a239: 6.88 fg: 5.29 ( cpu=5.42 (4m 58s 62ms) cpu:fg=5.29 (1m 12s 887ms) audio=1.46 (4m 53s 492ms) system_services=0.00414 )
# UID u0a239: 7.60 fg: 5.15 ( cpu=6.14 (5m 47s 183ms) cpu:fg=5.15 (1m 12s 191ms) audio=1.46 (4m 53s 867ms) system_services=0.00393 )
# UID u0a239: 7.75 fg: 4.97 ( cpu=6.28 (6m 5s 249ms) cpu:fg=4.97 (1m 11s 602ms) audio=1.46 (4m 53s 761ms) system_services=0.00446 )



# 8.49 CRNN_original
# UID u0a239: 7.24 fg: 5.41 ( cpu=5.78 (5m 12s 939ms) cpu:fg=5.41 (1m 15s 240ms) audio=1.46 (4m 53s 727ms) system_services=0.00330 )
# UID u0a239: 8.57 fg: 5.16 ( cpu=7.11 (6m 51s 897ms) cpu:fg=5.16 (1m 14s 118ms) audio=1.46 (4m 53s 387ms) system_services=0.00448 )
# UID u0a239: 9.67 fg: 5.12 ( cpu=8.21 (7m 54s 980ms) cpu:fg=5.12 (1m 12s 989ms) audio=1.46 (4m 53s 490ms) system_services=0.00509 )

# 7.13 CRNN_quant1
# UID u0a239: 6.94 fg: 5.15 ( cpu=5.47 (5m 14s 598ms) cpu:fg=5.15 (1m 15s 217ms) audio=1.46 (4m 53s 346ms) system_services=0.00360 )
# UID u0a239: 6.69 fg: 5.05 ( cpu=5.23 (5m 5s 652ms) cpu:fg=5.05 (1m 15s 128ms) audio=1.46 (4m 53s 381ms) system_services=0.00327 )
# UID u0a239: 7.75 fg: 4.97 ( cpu=6.29 (6m 5s 709ms) cpu:fg=4.97 (1m 11s 248ms) audio=1.46 (4m 53s 396ms) system_services=0.00403 )

# 7.18 CRNN_quant2.0
# UID u0a239: 7.21 fg: 5.48 ( cpu=5.75 (5m 12s 268ms) cpu:fg=5.48 (1m 15s 350ms) audio=1.46 (4m 53s 424ms) system_services=0.00315 )
# UID u0a239: 6.99 fg: 5.17 ( cpu=5.53 (5m 14s 375ms) cpu:fg=5.17 (1m 13s 794ms) audio=1.46 (4m 52s 925ms) system_services=0.00318 )
# UID u0a239: 7.33 fg: 5.10 ( cpu=5.87 (5m 39s 856ms) cpu:fg=5.10 (1m 14s 287ms) audio=1.46 (4m 53s 153ms) system_services=0.00360 )

# 7.62 CRNN_quant3
# UID u0a239: 7.33 fg: 5.27 ( cpu=5.87 (5m 23s 952ms) cpu:fg=5.27 (1m 16s 586ms) audio=1.46 (4m 53s 314ms) system_services=0.00321 )
# UID u0a239: 7.91 fg: 5.22 ( cpu=6.45 (6m 6s 206ms) cpu:fg=5.22 (1m 12s 600ms) audio=1.46 (4m 53s 367ms) system_services=0.00347 )
# UID u0a239: 7.61 fg: 5.86 ( cpu=6.14 (4m 41s 231ms) cpu:fg=5.86 (1m 6s 248ms) audio=1.46 (4m 53s 849ms) system_services=0.00371 )

# 7.45 CRNN_prune1_ex
# UID u0a239: 7.28 fg: 5.47 ( cpu=5.82 (5m 10s 739ms) cpu:fg=5.47 (1m 15s 309ms) audio=1.46 (4m 53s 155ms) system_services=0.00299 )
# UID u0a239: 7.84 fg: 5.12 ( cpu=6.38 (6m 4s 200ms) cpu:fg=5.12 (1m 12s 623ms) audio=1.46 (4m 53s 192ms) system_services=0.00383 )
# UID u0a239: 7.23 fg: 5.56 ( cpu=5.76 (4m 47s 533ms) cpu:fg=5.56 (1m 9s 437ms) audio=1.46 (4m 53s 525ms) system_services=0.00380 )

# 7.24 CRNN_prune2_ex
# UID u0a239: 7.05 fg: 5.29 ( cpu=5.59 (5m 9s 841ms) cpu:fg=5.29 (1m 15s 47ms) audio=1.46 (4m 53s 77ms) system_services=0.00332 )
# UID u0a239: 7.23 fg: 5.15 ( cpu=5.76 (5m 35s 552ms) cpu:fg=5.15 (1m 15s 491ms) audio=1.46 (4m 53s 117ms) system_services=0.00329 )
# UID u0a239: 7.44 fg: 5.12 ( cpu=5.98 (5m 42s 582ms) cpu:fg=5.12 (1m 13s 52ms) audio=1.46 (4m 53s 401ms) system_services=0.00340 )

