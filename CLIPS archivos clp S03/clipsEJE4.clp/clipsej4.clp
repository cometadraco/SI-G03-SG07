; Definicion de la plantilla Pais
(deftemplate Pais
  (slot Nombre)
  (multislot Bandera))

; Algunos ejemplos de paises con colores en sus banderas
(deffacts DatosPaises
  (Pais (Nombre Espania) (Bandera Rojo Amarillo))
  (Pais (Nombre Alemania) (Bandera Negro Rojo Amarillo))
  (Pais (Nombre Francia) (Bandera Azul Blanco Rojo))
  (Pais (Nombre Italia) (Bandera Verde Blanco Rojo))
  (Pais (Nombre Brasil) (Bandera Verde Amarillo Azul Blanco))
  (Pais (Nombre Peru) (Bandera Blanco Rojo))
  (Pais (Nombre Argentina) (Bandera Azul Blanco Amarillo))
)

; Regla para encontrar paises que tienen todos los colores especificados
(defrule BuscarColores
  ?b <- (buscar-colores $?colores)
  =>
  (bind ?lista-colores (create$ $?colores))
  (printout t "Buscando paises con los colores: " ?lista-colores crlf)
  (do-for-all-facts ((?p Pais)) TRUE
    (if (subsetp ?lista-colores (fact-slot-value ?p Bandera)) then
      (printout t "Pais: " (fact-slot-value ?p Nombre) crlf)))
  (retract ?b)
)

