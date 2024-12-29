import math
import os
import pickle
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy import stats
from scipy import interpolate
from glob import glob
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfilt
from config import cfg

# 讀取 CSV 檔案 (假設只有一列 "Gcode")
conbine = 0         #如果切硝訊號檔案不只一個填0，自動做合併
sr = cfg.DATASETS.SR           #sample rate
df = pd.read_csv(cfg.DATASETS.GCODE_PATH)  #gcode 位置
#訊號資料夾位置
filepaths = sorted(glob(cfg.DATASETS.DATA_PATHS))
if filepaths:
    filepath = filepaths[0]
    filename = os.path.basename(os.path.normpath(filepath))
    

loop_df = pd.DataFrame()

previous_x, previous_z, previous_f, previous_t, previous_RPM, recurrent_stride = 0.0, 0.0, 0.0, 0, 0.0, 0.0
max_s, base_rvc, fixed_rvc = 0.0, 0.0, 0.0
G_type, RPM_type = 0, 0
start_index = 0

def split_signal(signal):
    window_size = 1000
    stride = 1
    diff = []
    for i, start in enumerate(range(0+window_size, len(signal) - window_size + 1,stride)):
        end1 = start - window_size
        end2 = start + window_size
        segment1 = signal[end1:start]
        segment2 = signal[start:end2]
        mean_amp1 = np.mean(np.abs(segment1))  # 計算該段信號的平均振幅
        mean_amp2 = np.mean(np.abs(segment2))  # 計算該段信號的平均振幅
        diff.append(mean_amp2-mean_amp1)
    # 計算最大差距的兩個值
    if len(diff) > 1:
        # 使用 argsort 找出差距最大的兩個索引
        sorted_indices1 = np.argsort(diff)[-1]  # 取得兩個最大的差距的索引
        sorted_indices2 = np.argsort(diff)[0]
        
        print(f"這兩個差距出現的位置: {sorted_indices1+window_size}  {sorted_indices2+window_size}")
    return sorted_indices1*stride+window_size,sorted_indices2*stride+window_size

def segmentation(signal, lower_threshold ,segment_frame=None, segment_frame_feature=None):
    segment = np.where((signal > lower_threshold), 1, 0)
    
    if not segment_frame:
        segment_frame = list()
        segment_frame_feature = list()
    
    forward_state = 0
    temp_frame = []
    for i, j in enumerate(segment):
        if j == 0 & forward_state == 0:
            continue
        elif (j != 0) & (forward_state == 0):
            temp_frame.append(i)
        elif (j == 0) & (forward_state != 0):
            segment_frame.append(np.asarray(temp_frame))
            feature = np.array([len(temp_frame), signal[segment_frame[-1]].max()])
            segment_frame_feature.append(feature)
            temp_frame = []
        else:
            temp_frame.append(i)
        forward_state = j
    
    if (j != 0) & (forward_state != 0):
        segment_frame.append(np.asarray(temp_frame))
        feature = np.array([len(temp_frame), signal[segment_frame[-1]].max()])
        segment_frame_feature.append(feature)
        temp_frame = []
    return segment_frame, segment_frame_feature

def signal_segment(signals, cut_numbers,feed_rate = 11024):
    signals = abs(signals)
    
    # high pass filter
    # sos = butter(10, 10, 'hp', fs=sr, output='sos')
    # for i in range(1,3,1):
    #     signals[i] = sosfilt(sos, signals[i])
    
    # signal_square = np.square(signals[2])

    signal_square_average = signals#uniform_filter1d(signal_square, size=int(0.1 * sr), mode='constant', origin=int(((sr*0.1)/2)-1))
    
    select_frame = list()
    
    reduce_idx = 0
    for cut_number in zip(cut_numbers):
        segment_frame, segment_frame_feature = segmentation(signal=signal_square_average, lower_threshold=0.15)
        segment_frame_feature = np.stack(segment_frame_feature)
        for idx, feed_rate in zip(np.sort(np.argsort(segment_frame_feature[:, 0])[::-1][:sum(cut_numbers)])[len(select_frame):len(select_frame)+cut_number], [0.12]*cut_number):
            select_frame.append(segment_frame[idx])
            reduce_idx += 1
    
    return select_frame

def serch_loop(p,q,U,mode):
    global previous_x, previous_z,previous_f,previous_RPM,df,loop_df,start_index
    U = U*2
    # 尋找 N{p} 和 N{q} 的行索引
    nfirst_index = df[df['Gcode'].str.contains(f'N{p}')].index
    nsecond_index = df[df['Gcode'].str.contains(f'N{q}')].index

    # 檢查找到的索引
    if not nfirst_index.empty and not nsecond_index.empty:
        start_index = nfirst_index[0]  # N{p} 的索引
        end_index = nsecond_index[0]    # N{q} 的索引

        # 提取 N{p} 和 N{q} 之間的行
        relevant_lines = df.iloc[start_index:end_index + 1]

        # 儲存 X 和 Z 值的堆疊
        x_values = []
        z_values = []

        # 提取 X 和 Z 值
        for line in relevant_lines['Gcode']:
            x_match = re.search(r'X([-+]?\d*\.?\d+)', line)
            z_match = re.search(r'Z([-+]?\d*\.?\d+)', line)
            
            if x_match:
                x_values.append(float(x_match.group(1)))
            if z_match:
                z_values.append(float(z_match.group(1)))

    x_step = len(x_values)-1
    z_step = len(z_values)-1

    x_place = []
    z_place = []
    d_place = []
    t_place = []
    m = []

    while(1):
        if x_values[x_step] > x_values[0]:
            if previous_x - U >= x_values[x_step]:
                x_place.append(previous_x + U)
                z_place.append(z_values[x_step])
                x_place.append(previous_x + U)
                z_place.append(z_values[0])
            else:
                x_step = x_step-1
        else:
            if previous_x + U <= x_values[x_step]:
                
                x_place.append(previous_x + U)
                z_place.append(z_values[x_step])
                d_place.append(loop_distance(previous_x + U,z_values[x_step],previous_x,previous_z))
                t_place.append(loop_time_count(d_place[-1],previous_f,previous_RPM))
                m.append(mode)
                
                x_place.append(previous_x + U)
                z_place.append(z_values[0])
                d_place.append(None)
                t_place.append(None)
                m.append(mode)
                
                previous_x = previous_x + U
            else:
                x_step = x_step-1
        if x_step == 0:
            previous_z = z_values[0]
            break
    
    new_data = pd.DataFrame({
        'X': x_place,
        'Z': z_place,
        'D (mm)': d_place,
        'Time': t_place,
        'mode': m
    })
    loop_df = new_data

def loop_distance(x,z,previous_x,previous_z):
    return round(np.sqrt(((x - previous_x) / 2) ** 2 + (z - previous_z) ** 2)+0.05, 1) 

def loop_time_count(d,f,previous_RPM):
    return round((d/f)/(abs(previous_RPM)/60),4)   
  
def distance(parsed_data,previous_x,previous_z):
    return round(np.sqrt(((parsed_data['X'] - previous_x) / 2) ** 2 + (parsed_data['Z'] - previous_z) ** 2)+0.05, 1)    

def time_count(parsed_data,previous_RPM):
    return round((parsed_data['D (mm)']/parsed_data['F'])/(abs(parsed_data['RPM'])/60),4)   
                
def parse_gcode(gcode):
    global previous_x, previous_z,previous_f,previous_t,previous_RPM,max_s,base_rvc,fixed_rvc,G_type,RPM_type,recurrent_stride
    # 初始化字典來存儲結果
    parsed_data = {'X': None, 'Z': None, 'D (mm)': None, 'RPM': None, 'F':None, 'mode':None, 'Time':None}
            
    # 匹配X後的數值
    gcode = re.sub(r'\(.*?\)', '', gcode)
    x_match = re.search(r'X([+-]?\d+(\.\d+)?)', gcode)
    if x_match:
        parsed_data['X'] = float(x_match.group(1))
    else:
        parsed_data['X'] = previous_x
    
    # 匹配Z後的數值
    z_match = re.search(r'Z([+-]?\d+(\.\d+)?)', gcode)
    if z_match:
        parsed_data['Z'] = float(z_match.group(1))
    else:
        parsed_data['Z'] = previous_z
    
    # # U-X相對座標
    # u_match = re.search(r'U([+-]?\d+(\.\d+)?)', gcode)
    # if u_match:
    #     parsed_data['X'] = previous_x + float(u_match.group(1))
        
    # # W-Z相對座標
    # w_match = re.search(r'W([+-]?\d+(\.\d+)?)', gcode)
    # if w_match:
    #     parsed_data['Z'] = previous_z + float(w_match.group(1))

    if parsed_data['X'] is not None and parsed_data['Z'] is not None:
        if parsed_data['Z'] == previous_z:
            # 如果Z座標相等，則只考慮X座標的差值
            parsed_data['D (mm)'] = round(abs((parsed_data['X'] - previous_x) / 2)+0.05, 1)
        else:
            # 如果Z座標不相等，則使用完整的距離計算公式
            parsed_data['D (mm)'] = distance(parsed_data,previous_x,previous_z)
        previous_x = parsed_data['X']
        previous_z = parsed_data['Z']
    
    # 解析進給速度，假設是最後一個數值
    f_match = re.search(r'F([\d\.]+)', gcode)
    if f_match:
        parsed_data['F'] = previous_f = float(f_match.group(1))
    else:
        parsed_data['F'] = previous_f
        
        #切消指令
    g_match = re.search(r'G(?!99)([\d\.]+)', gcode)
    if g_match:
        g_value = int(g_match.group(1))
        if g_value == 0:        #快速移動
            G_type = 0
        if g_value == 1:        #直線切削
            G_type = 1
        if g_value == 3:        #以逆時針方向進行圓弧插補
            G_type = 3
        # if g_value == 4:        #停頓或延時指令
        #     G_type = 4
        if g_value == 50:       #最大轉速
            s_match = re.search(r'S([\d\.]+)', gcode)
            if s_match:
                max_s = float(s_match.group(1))
        if g_value == 71: 
            G_type = 71
            u_match = re.search(r'U([\d\.]+)', gcode)
            if u_match :
                recurrent_stride = float(u_match.group(1))
            p_match = re.search(r'P([\d\.]+)', gcode)
            q_match = re.search(r'Q([\d\.]+)', gcode)
            if p_match:
                serch_loop(int(p_match.group(1)),int(q_match.group(1)),recurrent_stride,previous_t)
        if g_value == 96:       #設定速度，會自動調整   r&vc
            s_match = re.search(r'S([\d\.]+)', gcode)
            if s_match:
                base_rvc = float(s_match.group(1))
            RPM_type = 96
        if g_value == 97:       #固定rpm
            s_match = re.search(r'S([\d\.]+)', gcode)
            if s_match:
                fixed_rvc = float(s_match.group(1))
            RPM_type = 97

    # 匹配T後的數值
    t_match = re.search(r'T([+-]?\d+(\.\d+)?)', gcode)
    if t_match:
        parsed_data['mode'] = int(t_match.group(1))
    else:
        parsed_data['mode'] = previous_t
    previous_t = parsed_data['mode']

    if parsed_data['X'] is not None and parsed_data['Z'] is not None and base_rvc!=0 and max_s!=0 and parsed_data['X'] !=0 :    #RPM_type = 96
        parsed_data['RPM'] = round(min((base_rvc * 1000) / (math.pi * parsed_data['X']), max_s), 0)
    if RPM_type == 97:      #RPM_type = 97
        parsed_data['RPM'] = fixed_rvc
    if G_type != 0 and G_type != 4 and parsed_data['X'] is not None and parsed_data['Z'] is not None and previous_RPM!=None:
        parsed_data['Time'] = time_count(parsed_data,previous_RPM)
        if parsed_data['Time'] <=0.003:
            parsed_data['Time'] = None
            
    previous_RPM = parsed_data['RPM']
    return pd.Series(parsed_data)

def synchronization(df_accelerometer, df_ServoGuide):
    interpolate_ServoGuide_signal = list()
    
    xnew = df_accelerometer.time.values
    for signal in df_ServoGuide.T.values[1:]:
        f = interpolate.interp1d(df_ServoGuide.time.values, signal, kind='nearest', fill_value="extrapolate")
        ynew = f(xnew)
        interpolate_ServoGuide_signal.append(ynew)    
    interpolate_ServoGuide_signal = np.stack(interpolate_ServoGuide_signal)
    
    crop_length = min(len(df_accelerometer), len(xnew))

    signals = np.concatenate([df_accelerometer.T.values[:,:crop_length], interpolate_ServoGuide_signal[:,:crop_length]], axis=0)
    return signals
            
def find_zSpeed_start_place(data,split_place,process_frame,sr,peaks):
    process_frame = int(process_frame)
    for i in peaks:
        if i > split_place+process_frame:   #找到的peak大於上個分割結束點
                if i-process_frame>=0:
                    # 檢查前面 process_frame 是否都在peak範圍內
                    if np.max(data[i-process_frame:i]) <= data[i] and np.max(data[i-process_frame+int(sr/10):i-process_frame+sr])<= data[i]/2:   #化新OP2 sr/5 other sr/10
                        return i-process_frame

# 使用 apply 對每一行的 Gcode 進行解析並新增對應的列
df[['X', 'Z', 'D (mm)','RPM','F','mode','Time']] = df.iloc[:,0].apply(parse_gcode)
if not loop_df.empty :
    df_before = df.iloc[:start_index ]  # 包含 `N{q}` 行
    df_after = df.iloc[start_index :]   # 從 `N{q}` 行之後的行
    df = pd.concat([df_before, loop_df, df_after], ignore_index=True)
    print('HAVE LOOP!!!')
# print(df)
# df.to_csv('parsed_output.csv')

bar = tqdm(filepaths, total=len(filepaths))
wrong_data = []
for filepath in bar:
    # try:
        print(filepath)
        filename = os.path.basename(filepath)
        print(filename)

        if conbine == 0:        #如果資料夾內的訊號是分開的會concat在一起
            df_accelerometer = pd.DataFrame()
            df_ServoGuide = pd.DataFrame()
            acc_last_time = 0
            servo_last_time = 0
            for num in range(int(len(os.listdir(filepath)) / 2)):
                # 讀取加速計資料
                accelerometer = pd.read_csv(
                    glob(f"{filepath}/Acc*-{num + 1}.csv")[0],
                    names=['time', 'spindle_front', 'turret'],
                    skiprows=[0],
                    low_memory=False,
                    index_col=None
                )

                # 讀取 ServoGuide 資料
                ServoGuide = pd.read_csv(
                    glob(f"{filepath}/Servo*-{num + 1}.csv")[0],
                    names=['time', 'motor_x_rpm', 'motor_x_current', 'motor_z_rpm', 'motor_z_current', 'spindle_rpm', 'spindle_current','S-TCDM'],
                    skiprows=[0],
                    low_memory=False,
                    index_col=None
                )
                # 若不是第一個檔案，調整時間軸以確保時間連續
                if num != 0:
                    accelerometer.iloc[:, 0] += acc_last_time
                    ServoGuide.iloc[:, 0] += acc_last_time

                # 更新最後一個時間點
                acc_last_time = accelerometer.iloc[-1, 0]
                # servo_last_time = ServoGuide.iloc[-1, 0]

                # 合併資料，這裡按行合併
                df_accelerometer = pd.concat([df_accelerometer, accelerometer],  axis=0)
                df_ServoGuide = pd.concat([df_ServoGuide, ServoGuide],  axis=0)
                
            df_ServoGuide = df_ServoGuide[['time', 'spindle_rpm', 'motor_x_rpm', 'motor_z_rpm', 'spindle_current', 'motor_x_current', 'motor_z_current','S-TCDM']]
        else:
            df_accelerometer = pd.read_csv(
                glob(f"{filepath}/Acc*-1.csv")[0],
                names=['time', 'spindle_front', 'turret'],
                skiprows=[0],
                low_memory=False
            )

            df_ServoGuide = pd.read_csv(
                glob(f"{filepath}/Servo*-1.csv")[0],
                names=['time', 'motor_x_rpm', 'motor_x_current', 'motor_z_rpm', 'motor_z_current', 'spindle_rpm', 'spindle_current'],
                skiprows=[0],
                low_memory=False,
            )
            df_ServoGuide = df_ServoGuide[['time', 'spindle_rpm', 'motor_x_rpm', 'motor_z_rpm', 'spindle_current', 'motor_x_current', 'motor_z_current']]   #motor_x_rpm=x-speed    motor_z_rpm=z-speed 

        signals = synchronization(df_accelerometer, df_ServoGuide)
        
        #存signal
        signals_list = []
        # with open('/data/Projects/quality-prediction/dataset/振鋒場域/{}/原始訊號/{}.pickle'.format(folder,filename),'wb') as file:
        #     signals_list.append(signals)
        #     pickle.dump(signals_list, file)

        z_speed = np.array(signals[5,:])
        x_speed = np.array(signals[4,:])
        times = np.array(signals[0,:])
        spindle_front = np.array(signals[1,:])
        spindle_rpm = np.array(signals[3,:])
        turret = np.array(signals[2,:])
        spindle_current = np.array(signals[6,:])
        motor_x_current = np.array(signals[7,:])
        motor_z_current = np.array(signals[8,:])

        spindle_front_MM = np.abs(spindle_front)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(spindle_front_MM.reshape(-1, 1))
        spindle_front_MM = scaler.transform(spindle_front_MM.reshape(-1, 1))
    

        turret =np.abs(turret)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(turret.reshape(-1, 1))
        turret = scaler.transform(turret.reshape(-1, 1))
        z_speed_MM = np.abs(z_speed)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(z_speed_MM.reshape(-1, 1))
        z_speed_MM = scaler.transform(z_speed_MM.reshape(-1, 1))
        x_speed_MM = np.abs(x_speed)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(x_speed_MM.reshape(-1, 1))
        x_speed_MM = scaler.transform(x_speed_MM.reshape(-1, 1))

        spindle_rpm_diff = np.abs(np.diff(spindle_rpm, prepend=0))
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(spindle_rpm_diff.reshape(-1, 1))
        spindle_rpm_diff = scaler.transform(spindle_rpm_diff.reshape(-1, 1))

        spindle_current = np.abs(spindle_current)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(spindle_current.reshape(-1, 1))
        spindle_current = scaler.transform(spindle_current.reshape(-1, 1))
        motor_x_current_MM = np.abs(motor_x_current)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(motor_x_current_MM.reshape(-1, 1))
        motor_x_current_MM = scaler.transform(motor_x_current_MM.reshape(-1, 1))
        motor_z_current_MM = np.abs(motor_z_current)
        scaler = MinMaxScaler(feature_range=(0, 1)).fit(motor_z_current_MM.reshape(-1, 1))
        motor_z_current_MM = scaler.transform(motor_z_current_MM.reshape(-1, 1))

        minmax_data = z_speed_MM + x_speed_MM +  motor_x_current_MM + motor_z_current_MM         
        
        low_threshold = np.median(minmax_data)
        print('low threshold :' + str(low_threshold))

        #取G_code中的可用時間
        df['Segment'] = (df['Time'].notna() & df['Time'].shift().isna()).cumsum()
        # 過濾掉包含NULL的行，只保留非NULL的行
        df_filtered = df.dropna(subset=['Time'])
        # 計算每段的總和
        time_sums = df_filtered.groupby('Segment')['Time'].sum()
        #找出此段的mode
        mode = df_filtered.groupby('Segment')['mode'].first() 
        for segment, time_sum in time_sums.items():
            print(f"Segment {segment} sum: {time_sum}")

        time_sums_array = time_sums.values

        mode = mode.values

        # time_sums_array[0] = 4

        start_array = []
        end_array = []
        mode_split_num = 0
        split_num = 0
        signal_N0_save = []
        signal_N1_save = []
        signal_N2_save = []
        signal_N3_save = []
        signal_N4_save = []
        minmax_data = minmax_data.ravel()   #攤平
        peaks, _ = find_peaks(minmax_data)

        for mode_num in range(len(mode)):             #找出每個mode的起迄點
            print(mode[mode_num])
            if mode_num == 0:
                split_place = 0
            else:
                split_place = end_array[mode_num-1]
            process_frame = int(sr*time_sums_array[mode_num])
            start_array.append(find_zSpeed_start_place(data=minmax_data,split_place = split_place,process_frame = process_frame,sr = sr,peaks = peaks))    #op1 frame_length 10000 、threshold 30
            if start_array[mode_num]==None:
                break
            end_array.append(int(start_array[mode_num]+process_frame))
            mode_split_num += 1

        
        print(start_array)
        print(end_array)
        # 畫圖
        plt.figure(figsize=(60, 20))
        plt.subplot(211)
        plt.plot(spindle_front)

        # 在標記的位置畫垂直線，並將兩兩一組的垂直線連接
        for i in range(0, len(start_array), 1):
        # 畫出有限高度的垂直線，上限是 max(spindle_front)
            plt.vlines(x=start_array[i], ymin=0, ymax=max(spindle_front), color='red', linestyle='--', label='Marked Position' if i == 0 else "")
            plt.vlines(x=end_array[i], ymin=0, ymax=max(spindle_front), color='red', linestyle='--')
            # 在頂部連接這兩條垂直線
            plt.plot([start_array[i], end_array[i]], [max(spindle_front), max(spindle_front)], color='red')

            midpoint = (start_array[i] + end_array[i]) / 2
            # 取對應的 mode 值
            mode_value = int(mode[i])
            # 在 position_array[i] 和 position_array[i+1] 之間顯示 mode
            plt.text(midpoint, max(spindle_front) * 1.02, f'T0{mode_value}', color='blue', ha='center', va='bottom', fontsize=18)
            
        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.title('Spindle_front')
        plt.legend()
        plt.grid(True)

        plt.subplot(212)

        plt.plot(minmax_data)
        for i in range(0, len(start_array), 1):
        # 畫出有限高度的垂直線，上限是 max(z_speed)
            plt.vlines(x=start_array[i], ymin=0, ymax=max(minmax_data), color='red', linestyle='--', label='Marked Position' if i == 0 else "")
            plt.vlines(x=end_array[i], ymin=0, ymax=max(minmax_data), color='red', linestyle='--')
            # 在頂部連接這兩條垂直線
            plt.plot([start_array[i], end_array[i]], [max(minmax_data), max(minmax_data)], color='red')

            midpoint = (start_array[i] + end_array[i]) / 2
            # 取對應的 mode 值
            mode_value = int(mode[i])
            # 在 position_array[i] 和 position_array[i+1] 之間顯示 mode
            plt.text(midpoint, max(minmax_data) * 1.02, f'T0{mode_value}', color='blue', ha='center', va='bottom', fontsize=18)

        plt.xlabel('Time')
        plt.ylabel('Signal Value')
        plt.title('total abs')
        plt.legend()

        # 保存圖像
        plt.tight_layout()
        # plt.savefig('/data/Projects/quality-prediction/dataset/振鋒場域/{}/切割訊號圖/{}.png'.format(folder,filename))
        # plt.savefig('/data/Projects/quality-prediction/dataset/ATRANS/客戶場域數據/OP2 (KB2-1141)/{}/切割訊號圖/{}.png'.format(folder,filename))
        plt.savefig(cfg.DATASETS.FIGURE_SAVE_PATH + filename + '.png')
        plt.close()
        exit()
        
    # except Exception as e:
    #     wrong_data.append(filename)
    #     continue
