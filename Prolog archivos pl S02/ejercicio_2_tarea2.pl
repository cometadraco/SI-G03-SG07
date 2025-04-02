horoscopo(aries, 21, 3, 19, 4).
horoscopo(tauro, 20, 4, 20, 5).
horoscopo(geminis, 21, 5, 20, 6).
horoscopo(cancer, 21, 6, 22, 7).
horoscopo(leon, 23, 7, 22, 8).
horoscopo(virgo, 23, 8, 22, 9).
horoscopo(libra, 23, 9, 22, 10).
horoscopo(escorpio, 23, 10, 21, 11).
horoscopo(sagitario, 22, 11, 21, 12).
horoscopo(capricornio, 22, 12, 19, 1).
horoscopo(acuario, 20, 1, 18, 2).
horoscopo(piscis, 19, 2, 20, 3).

% Regla para determinar si una fecha está dentro de un rango de fechas
esta_en_rango(Dia, Mes, Signo) :-
    horoscopo(Signo, DiaInicio, MesInicio, DiaFin, MesFin),
    (Mes > MesInicio; (Mes == MesInicio, Dia >= DiaInicio)),
    (Mes < MesFin; (Mes == MesFin, Dia =< DiaFin)).

% Regla para obtener el signo del Zodiaco para un día y mes dado
signo(Dia, Mes, Signo) :- esta_en_rango(Dia, Mes, Signo).
