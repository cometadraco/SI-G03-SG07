% Base de conocimientos sobre carreras
carrera(ingenieria_sistemas) :- tiene(matematicas), tiene(tecnologia).
carrera(medicina) :- tiene(biologia), tiene(salud).
carrera(derecho) :- tiene(argumentacion), tiene(justicia).
carrera(arquitectura) :- tiene(creatividad), tiene(diseno).
carrera(psicologia) :- tiene(empatia), tiene(comportamiento).
carrera(administracion_empresas) :- tiene(gestion), tiene(organizacion).
carrera(ingenieria_civil) :- tiene(matematicas), tiene(construccion).
carrera(ingenieria_industrial) :- tiene(matematicas), tiene(optimizacion).
carrera(economia) :- tiene(matematicas), tiene(analisis).
carrera(contabilidad) :- tiene(matematicas), tiene(finanzas).
carrera(disenio_grafico) :- tiene(creatividad), tiene(estetica).
carrera(marketing) :- tiene(estrategia), tiene(comunicacion).
carrera(biologia) :- tiene(biologia), tiene(laboratorio).
carrera(fisica) :- tiene(matematicas), tiene(experimentos).
carrera(quimica) :- tiene(experimentos), tiene(laboratorio).
carrera(educacion) :- tiene(ensenanza), tiene(paciencia).
carrera(turismo) :- tiene(idiomas), tiene(hospitalidad).
carrera(relaciones_internacionales) :- tiene(politica), tiene(negociacion).

% Preguntas al usuario
preguntar(Atributo, Pregunta) :-
    write(Pregunta), write(' (si./no.): '),
    read(Respuesta), nl,
    (Respuesta == si -> assertz(tiene(Atributo)) ; true).

% Evaluacion del usuario
evaluar :-
    preguntar(matematicas, 'Te gustan las matematicas?'),
    preguntar(biologia, 'Te gusta la biologia?'),
    preguntar(argumentacion, 'Tienes habilidades de argumentacion?'),
    preguntar(creatividad, 'Eres creativo?'),
    preguntar(tecnologia, 'Te interesa la tecnologia?'),
    preguntar(salud, 'Te interesa la salud?'),
    preguntar(justicia, 'Te interesa la justicia?'),
    preguntar(diseno, 'Te interesa el diseno?'),
    preguntar(empatia, 'Eres empatico?'),
    preguntar(comportamiento, 'Te interesa el comportamiento humano?'),
    preguntar(gestion, 'Te gusta la gestion?'),
    preguntar(organizacion, 'Te consideras organizado?'),
    preguntar(construccion, 'Te interesa la construccion?'),
    preguntar(optimizacion, 'Te gusta la optimizacion de procesos?'),
    preguntar(analisis, 'Te gusta el analisis de datos?'),
    preguntar(finanzas, 'Te interesa el mundo financiero?'),
    preguntar(estetica, 'Tienes sentido estetico?'),
    preguntar(estrategia, 'Te gusta diseniar estrategias?'),
    preguntar(comunicacion, 'Tienes habilidades de comunicacion?'),
    preguntar(laboratorio, 'Te gusta trabajar en un laboratorio?'),
    preguntar(experimentos, 'Disfrutas realizar experimentos?'),
    preguntar(ensenanza, 'Te gusta enseniar?'),
    preguntar(paciencia, 'Te consideras una persona paciente?'),
    preguntar(idiomas, 'Te gustan los idiomas?'),
    preguntar(hospitalidad, 'Te interesa la hospitalidad y el servicio?'),
    preguntar(politica, 'Te interesa la politica internacional?'),
    preguntar(negociacion, 'Te gusta la negociacion?'),
    recomendar.

% Generar recomendacion
recomendar :-
    carrera(Carrera),
    write('La carrera recomendada para ti es: '), write(Carrera), nl.

% Iniciar recomendacion
inicio :-
    write('Bienvenido al recomendador de carreras.'), nl,
    evaluar.
