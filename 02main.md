# 主程式
```
global loss #全域變數 loss：用於儲存當前的損失值，方便在不同函數中存取和更新
import time #用於計算程式執行時間

#主程式入口
if __name__ == '__main__':
    
    #f1:departure 車輛出發
    #f2:loading 裝貨               buffer-=1
    #f3:unload 卸貨加工            state+=1
    #f4:finish 加工完成            b,buffer+=1 s,state-=1       如果有挑到action 針對該貨物對 b-=1 s+=1
    #環境初始化
    env = Environment(40, 3, 12) #創建一個Environment實例，參數代表隨機種子、車輛數量和機台數量
    env.reset() #重置環境，初始化所有車輛和機台的狀態
    #print(env.event_list)
    #變數初始化
    k = day = 0 #k 和 day：用於追蹤時間和天數
    total=604800*2 #決定環境執行總時間14天
    tStart=time.time() #記錄程式開始執行的時間

#主模擬迴圈
    while env.event_list.Time[0] < total:
        ##畫圖
        for event in pg.event.get(): #pg.event.get()：獲取所有Pygame事件，如果有退出事件（如點擊關閉按鈕），則退出程式
            if event.type == pg.QUIT:
                exit()    
        env.add_event() #添加事件:根據當前環境狀態添加新的事件
        ##更新視覺化界面
        output_bg(env.machine_buffer,env.machine_state) #更新背景，顯示機台的緩衝區和工作狀態
        env_info(env.event_list.Time[0]) #顯示當前時間資訊
        truckV1_load(env.car_location)
        truckV2_load(env.car_location)
        truckV3_load(env.car_location) #更新並繪製三輛車的位置
        ##進度條與狀態顯示:使用print函數動態更新進度條，顯示當前進度百分比、損失值loss和機台緩衝區Br
        print('\r' + '[Progress]:[%s%s]%.2f%%;' % (
        '\033[1;32;43m🐵\033[0m' * int(env.event_list.Time[0]*20/total), '  ' * (20-int(env.event_list.Time[0]*20/total)),
        float(env.event_list.Time[0]/total*100))+'loss:%.3f'%loss+' Br:%s'%env.machine_buffer, end='')  
        
        env.add_event()
        ##生成新事件
        while env.event_list.Time[0] > (k * env.generation_rate): #根據生成速率generation_rate生成新的事件，並更新計數器k
            k += 1
            env.generate()
        ##處理不同類型的事件:根據事件類型（departure、loading、unload、finish）呼叫對應的方法來處理事件
        if env.event_list.type[0] == "departure":
            env.departure()
        elif env.event_list.type[0] == "loading":
            env.loading()
        elif env.event_list.type[0] == "unload":
            env.unload()
        elif env.event_list.type[0] == "finish":
            env.finish()
        ##事件列表排序與整理:對事件列表按時間排序，重置索引並刪除多餘的索引欄位，以保持事件列表的有序和整潔
        env.event_list = env.event_list.sort_values(axis=0, ascending=True, by=['Time'])  # 對時間做排序
        env.event_list = env.event_list.reset_index()  # 調整index
        env.event_list = env.event_list.drop('index', axis=1)  # 把多於的刪除
        
        #每一段時間記錄一次工單量，也會記錄最佳模型(每日記錄與模型儲存)
        ##每日記錄
        count_day = env.event_list.Time[0] / 43200  #每43200秒(12小時)記錄一次出貨量
        if count_day > day+1:     #當count_day超過day + 1時，表示進入了新的一天，需要進行記錄和模型儲存    
            print(env.machine_buffer[11])
            ##記錄機台緩衝區:env.machine_buffer[11]：記錄特定機台（第12台）的緩衝區數量，並添加到day_list列表中
            day_list.append(env.machine_buffer[11])
            ##模型儲存
            if day == 1 : #第一天：計算第一天的緩衝區增長量，並將模型儲存為Best_Model.pt
                print(day_list[day]-day_list[day-1])
                max_day = day_list[day]-day_list[day-1] #過歷史最大緩衝區增長量
                torch.save(net,'Best_Model.pt')
                print("Saved Model..")
            if day > 1 : #後續天數：比較當前天數的緩衝區增長量是否超過歷史最大值max_day，如果超過，則更新max_day並儲存模型
                print(day_list[day]-day_list[day-1])
                compare = day_list[day]-day_list[day-1]
                if compare > max_day :
                    max_day = compare
                    torch.save(net,'Best_Model.pt')
                    print("Saved Model..")
            ##更新天數：day += 1，準備進入下一天的記錄
            day += 1
        #clock.tick(10) #可調慢程式執行速度 (觀察畫圖用)
    ##結束Pygame：當模擬時間達到總時間total後，退出Pygame視窗
    pg.quit()        

#程式結束與結果輸出
##計算總執行時間
tEnd=time.time() #記錄程式結束時的時間
#torch.save(net,'Best_Model.pt') 
print("Spent Time :",tEnd-tStart) #程式總執行時間(s)，計算並輸出程式總共花費的時間（秒）
```
```
##計算演算法結果(DQN)
DQN_result=0 #DQN_result被初始化為0，下一段被註解掉的程式碼用於計算DQN的結果
# for i in range(len(action_choose)):
#     if action_choose[i]==24 or action_choose[i]==25:
#         DQN_result=DQN_result+1
##計算演算法結果(GCN)
gcn_result=0 #gcn_result被初始化為0，通過遍歷w_action_choose列表，統計選擇了動作編號23或24的次數
for i in range(len(w_action_choose)):
    if w_action_choose[i]==23 or w_action_choose[i]==24:
        gcn_result=gcn_result+1

print("傳統演算法_result",DQN_result)
print("gcn_result",gcn_result)#5100 #5411  ##ATTENTION 5187 5328 #ATTENTION with/d_k 5589 #輸出GCN的結果
##輸出機台緩衝區和每日變化
print("BR",env.machine_buffer) #輸出所有機台的緩衝區狀態
```
```
print(day_list) #輸出每日記錄的機台緩衝區數量
for i in range(len(day_list)-1):
    print(day_list[i+1]-day_list[i]) #遍歷day_list，計算並輸出每一天的緩衝區增長量
```
# PLOT
```
#繪製損失隨著回合數變化的曲線，並將損失數據保存到文本文件，以便未來分析和比較: DQN 和 GCN
##使用matplotlib將Loss_plt列表中的損失值繪製成曲線圖，顏色設為紫色，標籤為'Loss'
plt.plot(np.array(Loss_plt), c='purple', label='Loss')
plt.legend(loc='best')
plt.ylabel('Loss')
plt.xlabel('episodes')
##添加圖例、標籤和坐標軸範圍（0到70000回合，損失值範圍0到100）
plt.axis([0, 70000, 0, 100])
##保存損失數據
np.savetxt('Loss.txt',Loss_plt,fmt="%f" ) #使用np.savetxt將損失值列表Loss_plt保存到Loss.txt文件中，格式為浮點數
plt.grid()
plt.show() #使用plt.show()顯示損失值曲線圖
```
