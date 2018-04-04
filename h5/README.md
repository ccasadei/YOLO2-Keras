I pesi di **BASE** sono scaricabili da qui: 
https://1drv.ms/f/s!ApLdDEW3ut5fec2OzK4S4RpT-SU

Selezionare il file `full_yolo_backend.h5` e rinominarlo in `base_backend.h5`.

Il file `base.h5` se esiste è quello che contiene i pesi di backend e frontend allenati su COCO.

Il file `pretrained.h5` se esiste è quello che viene caricato all'inizio del training come partenza (è impostato sulle classi indicate in `configYOLO.json`).

Il file `result.h5` se esiste è quello creato alla fine dell'allenamento.

Il file `chkpnt_best.h5` se esiste è quello creato durante l'allenamento come checkpoint migliore. Può essere rinominato in `pretrained.h5` se si interrompe l'allenamento e successivamente si vuole ripartire da quel punto.

Tutti i nomi ed i path dei file sono comunque configurabili da `configYOLO.json`.


