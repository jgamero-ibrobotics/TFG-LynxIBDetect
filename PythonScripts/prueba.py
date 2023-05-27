import numpy as np

# Definir la matriz
matriz = [[0.8, 0.7, 0.0, 0.0, 0.4],
                   [0.9, 0.8, 0.6, 0.1, 0.0],
                   [0.0, 0.4, 0.7, 0.0, 0.0],]



def elimina_no_maximos(matriz, col, indice_preservar):
  
    for i in range(len(matriz[:,col])):
        if i != indice_preservar:
            matriz[i,col] = 0.0
    return matriz

if True:
    for i in range(5):
        if i < 3:
            continue
        else:
            matriz.append([0, 0, 0, 0, 0])

matriz = np.array(matriz)
print(matriz)
iou_max_score = []
for col in range(5):
    column = matriz[:,col]
    iou_max_index_col  = np.argmax(column)

    row = matriz[iou_max_index_col,:]
    iou_max_index_row  = np.argmax(row)

    if [iou_max_index_col, iou_max_index_row] == [iou_max_index_col, col]: # si el maximo IoU de la columna es el mismo que el maximo IoU de la fila
        iou_max_score.append(column[iou_max_index_col])
        matriz = elimina_no_maximos(matriz, col, iou_max_index_col)
    else:
        while [iou_max_index_col, iou_max_index_row] != [iou_max_index_col, col]:

            if  np.all(row == 0.0) and np.all(column == 0.0):
                 break
            elif np.any(row != 0.0) and np.all(column == 0.0):
                iou_max_index_col = iou_max_index_col + 1
                row = matriz[iou_max_index_col,:]
            else:
                column[iou_max_index_col] = 0.0
                iou_max_index_col  = np.argmax(column)

                row = matriz[iou_max_index_col,:]
                iou_max_index_row  = np.argmax(row)


        matriz = elimina_no_maximos(matriz, col, iou_max_index_col)
        iou_max_score.append(column[iou_max_index_col])

print(iou_max_score)