best_ins_d_all: lista di liste. 
   0: numero del veicolo
   1: [prev_n, next_n, node_id]
   2: valore di c1 oppure c2 (costo associato a questo inserimento)



- ins_avail_flag = true
- while there are still insertions that can be made (flag ins_avail_flag):
   - per ogni nodo d tale che d['chosen_vrp']==false e d['crowdshipped']==false:
      - per ogni veicolo k tale che d['vol'] < sol[k]['vol_left']:
         - trova il migliore inserimento di d in k tra quelli time feasible (in termini di c1)
         - inseriscilo in best_ins_d_all
      - ordina best_ins_d_all in base al costo
      - calcola il regret-k value di d
      - se il regret value di d è più alto del miglior regret value trovato:
         - sostituisci il miglior regret value trovato
         - sostituisci il miglior inserimento (deepcopy?)
   - effettua il miglior inserimento, se esistente
   - se non è esistente, set ins_avail_flag=false
         

         
         ins_c = self.getC3(sol[k],[])