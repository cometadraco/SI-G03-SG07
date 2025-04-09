(deffacts datos
   (cadena1 B C A D E E B C E)
   (cadena2 E E B C A F E))
(deftemplate union
   (slot letra))

(defrule unir-letras-comunes
   ?c1 <- (cadena1 $?inicio ?letra $?fin)
   (cadena2 $?inicio2 ?letra $?fin2)
   (not (union (letra ?letra))) ;; Evita duplicados
   =>
   (assert (union (letra ?letra))))