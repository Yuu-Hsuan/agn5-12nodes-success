# 主程式
```
global loss
import time
if __name__ == '__main__':
    
    #f1:departure 車輛出發
    #f2:loading 裝貨               buffer-=1
    #f3:unload 卸貨加工            state+=1
    #f4:finish 加工完成            b,buffer+=1 s,state-=1       如果有挑到action 針對該貨物對 b-=1 s+=1
    env = Environment(40, 3, 12)
    env.reset()
    #print(env.event_list)
    k = day = 0
    total=604800*2 #決定環境執行總時間
    tStart=time.time()
    while env.event_list.Time[0] < total:
        #畫圖
        for event in pg.event.get():
            if event.type == pg.QUIT:
                exit()    
        env.add_event()
        output_bg(env.machine_buffer,env.machine_state)
        env_info(env.event_list.Time[0])
        truckV1_load(env.car_location)
        truckV2_load(env.car_location)
        truckV3_load(env.car_location)
        print('\r' + '[Progress]:[%s%s]%.2f%%;' % (
        '\033[1;32;43m🐵\033[0m' * int(env.event_list.Time[0]*20/total), '  ' * (20-int(env.event_list.Time[0]*20/total)),
        float(env.event_list.Time[0]/total*100))+'loss:%.3f'%loss+' Br:%s'%env.machine_buffer, end='')  
        
        env.add_event()
        
        while env.event_list.Time[0] > (k * env.generation_rate):
            k += 1
            env.generate()
        if env.event_list.type[0] == "departure":
            env.departure()
        elif env.event_list.type[0] == "loading":
            env.loading()
        elif env.event_list.type[0] == "unload":
            env.unload()
        elif env.event_list.type[0] == "finish":
            env.finish()

        env.event_list = env.event_list.sort_values(axis=0, ascending=True, by=['Time'])  # 對時間做排序
        env.event_list = env.event_list.reset_index()  # 調整index
        env.event_list = env.event_list.drop('index', axis=1)  # 把多於的刪除
        
        #每一段時間記錄一次工單量，也會記錄最佳模型
        count_day = env.event_list.Time[0] / 43200  #每43200秒(12小時)記錄一次出貨量
        if count_day > day+1:        
            print(env.machine_buffer[11])
            day_list.append(env.machine_buffer[11])
            if day == 1 :
                print(day_list[day]-day_list[day-1])
                max_day = day_list[day]-day_list[day-1]
                torch.save(net,'Best_Model.pt')
                print("Saved Model..")
            if day > 1 :
                print(day_list[day]-day_list[day-1])
                compare = day_list[day]-day_list[day-1]
                if compare > max_day :
                    max_day = compare
                    torch.save(net,'Best_Model.pt')
                    print("Saved Model..")
            day += 1
        #clock.tick(10) #可調慢程式執行速度 (觀察畫圖用)  
    pg.quit()        

tEnd=time.time()
#torch.save(net,'Best_Model.pt')
print("Spent Time :",tEnd-tStart) #程式總執行時間(s)
```
```
DQN_result=0
# for i in range(len(action_choose)):
#     if action_choose[i]==24 or action_choose[i]==25:
#         DQN_result=DQN_result+1
gcn_result=0
for i in range(len(w_action_choose)):
    if w_action_choose[i]==23 or w_action_choose[i]==24:
        gcn_result=gcn_result+1

print("傳統演算法_result",DQN_result)
print("gcn_result",gcn_result)#5100 #5411  ##ATTENTION 5187 5328 #ATTENTION with/d_k 5589
print("BR",env.machine_buffer)
```
```
print(day_list)
for i in range(len(day_list)-1):
    print(day_list[i+1]-day_list[i])
```
# PLOT
```
#繪製損失隨著回合數變化的曲線，並將損失數據保存到文本文件，以便未來分析和比較: DQN 和 GCN
plt.plot(np.array(Loss_plt), c='purple', label='Loss')
plt.legend(loc='best')
plt.ylabel('Loss')
plt.xlabel('episodes')
plt.axis([0, 70000, 0, 100])
np.savetxt('Loss.txt',Loss_plt,fmt="%f" )
plt.grid()
plt.show()
```
