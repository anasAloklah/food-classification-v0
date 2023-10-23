def ListToString(list,moreTabe=False):
    res = ''
    for i in list:
        res = res + '\t' + str(i)

    return res


days=['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
res=input('Do you want to enter data :')
total=[]
listoftotal=[]
while(res=='y' or res=='Y'):
    print('Enter the store’s sales for each day')
    for i in range(0,len(days)):
        x=int(input(days[i]+': '))
        total.append(x)

    print('The total sales for the week: ',sum(total) )
    print('The average daily sales: ', round(sum(total)/len(total),2))
    print('The highest amount is ', max(total))
    print('The lowest amount is: ', min(total))

    ave=round(float(sum(total)/len(total)),2)
    total.append(sum(total))
    total.append(ave)
    total.append(max(total))
    total.append(min(total))
    res = input('Do you want to enter data :')
    listoftotal.append(total)
    total=[]
days.append('Total')
days.append('average')
days.append('Maximum')
days.append('Minimum')
f=open('sale.txt','w')
f.write(ListToString(days))
f.write('\n')
for list in listoftotal:
    f.write(ListToString(list))
    f.write('\n')
f.close()

