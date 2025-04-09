(deftemplate numero
   (slot valor))

(deftemplate minimo
   (slot valor))

; Datos de entrada con números a evaluar
(deffacts datos-prueba
   (numero (valor 8))
   (numero (valor 2))
   (numero (valor 15))
   (numero (valor 4))
   (numero (valor 7)))

; Regla para establecer el primer número como el mínimo
(defrule inicializar-minimo
   ?n <- (numero (valor ?v))
   (not (minimo (valor ?)))
   =>
   (assert (minimo (valor ?v))))  ; Inicializa el mínimo con el primer número

; Regla para actualizar el mínimo si se encuentra un número menor
(defrule actualizar-minimo
   ?n <- (numero (valor ?v1))
   ?m <- (minimo (valor ?v2))
   (test (< ?v1 ?v2))  ; Si encontramos un valor menor que el mínimo actual
   =>
   (retract ?m)  ; Retira el mínimo anterior
   (assert (minimo (valor ?v1))))  ; Inserta el nuevo mínimo

; Regla para imprimir el valor mínimo encontrado
(defrule mostrar-minimo
   (minimo (valor ?v))
   =>
   (printout t crlf "El valor mínimo encontrado es: " ?v crlf))  ; Imprime el valor mínimo final
