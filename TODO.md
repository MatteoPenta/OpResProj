VRP
- Calcola distance matrix
- Ordina ogni riga della distance matrix, crescente
- Crea lista di nodi con valore binario "preso"
- Per ogni camion
   - Aggiungo il nodo 0 (deposito)
   - Aggiungo il primo nodo non preso più vicino al deposito
   - feasible_nodes_flag = 1
   - while (feasible_nodes_flag)
      - feasible_nodes_flag = 0
      - Itero in ogni nuovo nodo non ancora collegato
         - inizializzo minimo {i_pm1, u, i_p, c1_min} (u: current node)
         - Considero ogni coppia di nodi nella current solution
            - Se inserire il nuovo nodo non collegato è feasible
               - feasible_nodes_flag = 1
               - Calcolo il costo c1
               - Se c1 < c1_min
                  - Aggiorno {i_pm1, u, i_p, c1_min}
         - Aggiungo {i_pm1, u, i_p, c1_min} alla lista best_placing (un elemento per nodo libero)
      - inizializzo optimum c2 
      - Itero nella lista best_placing SE feasible_nodes_flag==1
         - Calcolo c2 per l'elemento corrente
         - se c2 è meglio di c2_opt
            - aggiorno c2_opt
      - Se feasible_nodes_flag==1 
         - Aggiungo il nodo corrispondente a c2_opt alla soluzione nella posizione giusta
         - Aggiorno il tempo di arrivo a tutti i nodi nella soluzione dal nodo nuovo all'ultimo (sono i valori b_j)

   - Aggiungo il nodo 0 (deposito)