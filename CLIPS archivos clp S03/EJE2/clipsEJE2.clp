(deftemplate numero
   (slot valor)  ; Define el campo 'valor' para almacenar el número
)

(deffacts lista-de-enteros
   (numero (valor 1))
   (numero (valor 2))
   (numero (valor 3))
   (numero (valor 4))
   (numero (valor 5))
)

(defrule sumar-elementos
   =>
   (bind ?suma 0)  ; Inicializa la suma en 0

   ; Itera sobre todos los hechos de tipo 'numero'
   (do-for-all-facts ((?facto numero)) TRUE
      (bind ?valor (fact-slot-value ?facto valor))  ; Obtiene el valor del campo 'valor'
      (bind ?suma (+ ?suma ?valor))  ; Suma el valor al total
   )

   (printout t "La suma de los elementos es: " ?suma crlf)
)

(reset)
(run)
