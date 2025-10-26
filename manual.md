# Benutzerhandbuch: HPIO Control Center

_Willkommen beim HPIO Control Center, einer leistungsstarken Software zur Optimierung komplexer Probleme mittels verschiedener Algorithmen. Diese Anwendung implementiert einen **H**ybrid **P**heromone **I**nspired **O**ptimizer (HPIO), der von der Natur inspiriert ist, um optimale Lösungen für mathematische Zielfunktionen zu finden. Darüber hinaus bietet sie Vergleichsmöglichkeiten mit klassischen Optimierungsalgorithmen wie Differential Evolution, Partikel-Schwarm-Optimierung und Genetischen Algorithmen.

Das HPIO Control Center löst das Problem der Suche nach den globalen Minima von mehrdimensionalen Funktionen, die in vielen wissenschaftlichen und technischen Bereichen auftreten. Es bietet eine interaktive Oberfläche zur Visualisierung des Optimierungsprozesses und zur Feinabstimmung der Algorithmusparameter, um die bestmöglichen Ergebnisse zu erzielen._

## 1. Erste Schritte: Installation und Start

Das HPIO Control Center ist eine Python-Anwendung, die über Streamlit ausgeführt wird. Um die Anwendung zu starten, müssen Sie Python und die erforderlichen Bibliotheken installiert haben.

**1.1 Voraussetzungen**
*   **Python 3.8+**: Stellen Sie sicher, dass Python auf Ihrem System installiert ist.
*   **Paketmanager pip**: Dieser ist normalerweise zusammen mit Python installiert.

**1.2 Installation der Abhängigkeiten**
Öffnen Sie ein Terminal oder eine Eingabeaufforderung und navigieren Sie in das Verzeichnis, in dem sich die Dateien `hpio.py`, `hpio_record.py` und `streamlit_app.py` befinden. Führen Sie dann den folgenden Befehl aus, um alle benötigten Bibliotheken zu installieren:

```bash
pip install numpy pandas streamlit matplotlib imageio
```

Für die optionale GPU-Beschleunigung (PyOpenCL) und erweiterte Videoexport-Funktionen (FFmpeg/OpenCV) können zusätzliche Installationen erforderlich sein:

```bash
pip install pyopencl opencv-python
```

**1.3 Starten der Anwendung**
Nachdem alle Abhängigkeiten installiert sind, starten Sie die Anwendung über das Terminal im selben Verzeichnis:

```bash
streamlit run streamlit_app.py
```

Die Anwendung wird in Ihrem Standard-Webbrowser geöffnet. Sollte dies nicht automatisch geschehen, wird im Terminal eine URL angezeigt, die Sie manuell öffnen können (z.B. `http://localhost:8501`).

## 2. Die Benutzeroberfläche (GUI) im Überblick

Das HPIO Control Center ist in mehrere Seiten unterteilt, die über eine Navigationsleiste auf der linken Seite zugänglich sind. Jede Seite dient einem spezifischen Zweck:

*   **Seitenleiste (Navigation)**: Hier wählen Sie die gewünschte Seite aus (z.B. 'Start / Run', 'Parameter', 'Algorithmen', 'Presets', 'Aufnahme & Export', 'Experimente', 'Hilfe').
*   **Hauptbereich**: Zeigt den Inhalt der aktuell ausgewählten Seite an.

**2.1 Seitenleiste**
Die Seitenleiste (links) enthält die Hauptnavigation und je nach ausgewählter Seite auch spezifische Steuerelemente und Einstellungen.

**2.2 Hauptbereich**
Der Hauptbereich ist dynamisch und ändert sich je nach der in der Seitenleiste ausgewählten Seite. Er ist typischerweise in mehrere Abschnitte unterteilt, die durch Überschriften (`###`) und manchmal durch Spalten (`st.columns`) oder Tabs (`st.tabs`) organisiert sind.

**2.3 Allgemeine GUI-Elemente**
*   **Selectboxen**: Ermöglichen die Auswahl aus vordefinierten Optionen (z.B. 'Zielfunktion').
*   **Number Inputs**: Für die Eingabe numerischer Werte (z.B. 'Iterationen', 'Seed').
*   **Slider**: Für die Feinabstimmung numerischer Werte innerhalb eines Bereichs (z.B. 'Traillänge', 'Momentum').
*   **Checkboxes**: Zum Aktivieren oder Deaktivieren von Optionen (z.B. 'GPU (PyOpenCL)', 'Overlay').
*   **Buttons**: Zum Auslösen von Aktionen (z.B. 'Start', 'Pause', 'Stop', 'Übernehmen').
*   **Text Inputs**: Für die Eingabe von Text (z.B. 'Dateiname').
*   **Formulare**: Gruppieren mehrere Eingabefelder und erfordern einen 'Übernehmen'-Button, um Änderungen anzuwenden.
*   **Diagramme und Bilder**: Zeigen Visualisierungen des Optimierungsprozesses oder Ergebnisse an.

## 3. HPIO-Optimierung starten und steuern ('Start / Run')

Die Seite 'Start / Run' ist das Herzstück der Anwendung, wo Sie die HPIO-Optimierung konfigurieren, starten und live verfolgen können.

**3.1 Basis-Setup**
In der Seitenleiste finden Sie die grundlegenden Einstellungen:
*   **Zielfunktion**: Wählen Sie eine der vordefinierten Funktionen aus: `rastrigin`, `ackley` oder `himmelblau`. Die Auswahl einer neuen Zielfunktion setzt die Parameter auf die Standardwerte für diese Funktion zurück und markiert den Lauf als 'Parameter geändert'.
*   **GPU (PyOpenCL)**: Aktivieren Sie diese Option, um die Berechnung auf einer kompatiblen GPU zu beschleunigen, falls PyOpenCL installiert und ein Gerät verfügbar ist. Andernfalls läuft die Optimierung auf der CPU.

**3.2 Seed & Iterationen**
*   **Seed**: Eine ganze Zahl, die den Startwert des Zufallszahlengenerators festlegt. Ein fester Seed sorgt für reproduzierbare Ergebnisse. Klicken Sie auf den '🎲 Zufalls-Seed'-Button, um einen neuen, zufälligen Seed zu generieren.
*   **Iterationen**: Die maximale Anzahl der Optimierungsschritte, die der Algorithmus ausführen soll.

**3.3 Visualisierung**
*   **Viz-Frequenz**: Legt fest, wie oft (jede n-te Iteration) die Heatmap und die Agentenpositionen aktualisiert werden. Eine höhere Frequenz kann die Anwendung verlangsamen.
*   **Overlay (Iteration / Bestwert)**: Zeigt die aktuelle Iterationsnummer und den besten gefundenen Wert direkt in der Heatmap-Visualisierung an.
*   **Traillänge**: Bestimmt, wie viele vorherige Positionen der Agenten als 'Spur' in der Heatmap angezeigt werden. Längere Spuren zeigen den Bewegungspfad besser, können aber die Performance beeinflussen.

**3.4 Run-Kontrollen**
Diese Buttons steuern den Optimierungsprozess:
*   **Start**: Initialisiert und startet einen neuen Optimierungslauf mit den aktuellen Parametern. Dies setzt alle vorherigen Ergebnisse und die Historie zurück.
*   **Pause / Weiter**: Unterbricht den laufenden Optimierungsprozess oder setzt ihn fort.
*   **Stop**: Beendet den aktuellen Optimierungslauf vollständig.
*   **Schritt vor**: Führt nur einen einzelnen Optimierungsschritt aus. Nützlich für die detaillierte Analyse oder wenn der Lauf pausiert ist.
*   **Reset**: Setzt den aktuellen Lauf zurück, behält aber den Seed bei. Die Agenten werden neu initialisiert.
*   **Reset + neuer Seed**: Setzt den Lauf zurück und generiert einen neuen, zufälligen Seed.

**3.5 Live-Parameteranpassung**
Unter dem Expander '🔄 Live-Parameteranpassung' können Sie wichtige Agentenparameter während eines laufenden Optimierungsprozesses anpassen, ohne den Lauf neu starten zu müssen:
*   **Agent step**: Die Schrittgröße der Agenten.
*   **Curiosity**: Der Grad der zufälligen Erkundung der Agenten.
*   **Momentum**: Die Trägheit der Agentenbewegung.
*   **deposit_sigma**: Die Breite des Fußabdrucks, den ein Agent im Feld hinterlässt.
*   **coherence_gain**: Die Stärke, mit der Agenten in Richtung des global besten gefundenen Punktes gezogen werden.
*   **w_intensity / w_phase / phase_span_pi**: Parameter, die die Art und Weise beeinflussen, wie Agenten Spuren im Feld deponieren, insbesondere in Bezug auf Intensität und Phaseninformationen.
*   **Overlay anzeigen**: Schaltet das Overlay in der Heatmap live um.

Klicken Sie auf **'Änderungen anwenden'**, um die angepassten Werte sofort im laufenden Optimierungsprozess zu übernehmen.

**3.6 Hauptanzeige**
Der Hauptbereich zeigt die Live-Visualisierung und Statusinformationen:
*   **Heatmap & Agents**: Zeigt eine Heatmap des Feldes (logarithmisch skaliert) und die aktuellen Positionen der Agenten sowie deren Spuren. Die Farbe der Heatmap (Standard: inferno) zeigt die Intensität des Pheromonfeldes an.
*   **Parameter Snapshot**: Eine JSON-Darstellung der aktuell verwendeten Konfiguration.
*   **Status**: Zeigt die aktuelle Iteration, den besten gefundenen Wert, die beste Position und die Verbesserung seit dem letzten Schritt an. Auch die Zeit pro Iteration und die Gesamtzeit werden hier angezeigt.
*   **Konsole**: Ein Log-Bereich, der wichtige Meldungen und den Fortschritt des Optimierungslaufs anzeigt.
*   **Live-Metriken**: Ein Liniendiagramm, das die Entwicklung des besten gefundenen Wertes über die Iterationen hinweg darstellt, sowie weitere Metriken wie Δ Best und Iterationen.

## 4. Detaillierte Funktionsbeschreibung: Parameter ('Parameter')

Die Seite 'Parameter' ermöglicht die detaillierte Konfiguration aller HPIO-Algorithmusparameter. Diese Einstellungen werden wirksam, wenn ein neuer Lauf gestartet oder ein Reset durchgeführt wird.

**4.1 Feldparameter**
Diese Parameter beeinflussen das Verhalten des Pheromonfeldes:
*   **Grid Breite / Grid Höhe**: Die Dimensionen des internen Rasters, auf dem das Pheromonfeld berechnet wird. Größere Grids bieten mehr Detail, können aber die Rechenzeit erhöhen.
*   **relax_alpha**: Die Stärke der Glättung des Feldes. Höhere Werte führen zu einer stärkeren Glättung, niedrigere Werte zu schärferen Pheromon-Peaks.
*   **evap**: Die Verdunstungsrate des Pheromons. Höhere Werte lassen alte Spuren schneller verschwinden.
*   **kernel_sigma**: Die Standardabweichung des Gaußschen Kernels, der für die Feldglättung verwendet wird. Beeinflusst die 'Breite' der Pheromonspuren.

**4.2 Agenten & Ablageparameter**
Diese Parameter steuern das Verhalten der Agenten und ihre Interaktion mit dem Feld:
*   **count**: Die Anzahl der Agenten im Schwarm.
*   **step**: Die grundlegende Schrittgröße, mit der sich die Agenten bewegen.
*   **curiosity**: Ein Faktor, der die zufällige Bewegung der Agenten beeinflusst, um neue Bereiche zu erkunden.
*   **momentum**: Die Trägheit der Agentenbewegung. Höhere Werte lassen Agenten ihre aktuelle Bewegungsrichtung länger beibehalten.
*   **deposit_sigma**: Die Breite des Gaußschen Fußabdrucks, den ein Agent bei der Ablage von Pheromonen hinterlässt. Ein Wert von 0.0 bedeutet eine Punktablage.
*   **coherence_gain**: Ein Faktor, der die Anziehung der Agenten zum global besten gefundenen Punkt steuert. Hilft dem Schwarm, sich auf vielversprechende Bereiche zu konzentrieren.
*   **w_intensity**: Gewichtung des Intensitätsanteils bei der Pheromonablage. Bessere Agenten legen stärker ab.
*   **w_phase**: Gewichtung des Phasenanteils bei der Pheromonablage. Eine leichte Modulation basierend auf der Bewegungsrichtung.
*   **phase_span_pi**: Die Spannweite der Phasenmodulation in Einheiten von π.

**4.3 Annealing-Parameter**
Diese Parameter steuern die dynamische Anpassung von `step` und `curiosity` über die Laufzeit des Algorithmus hinweg:
*   **anneal_step_from / anneal_step_to**: Start- und Endwert für die lineare Interpolation der Schrittgröße.
*   **anneal_curiosity_from / anneal_curiosity_to**: Start- und Endwert für die lineare Interpolation der Neugier.

**4.4 Frühabbruch & Polish**
*   **early_patience**: Anzahl der Iterationen ohne signifikante Verbesserung, bevor der Lauf vorzeitig beendet wird.
*   **early_tol**: Die minimale Verbesserung, die als 'signifikant' gilt, um den Frühabbruch-Zähler zurückzusetzen.
*   **polish_h**: Der Schrittparameter für die lokale quadratische Nachbesserung des besten gefundenen Punktes am Ende des Laufs.

**4.5 Änderungen anwenden**
Nachdem Sie Parameter in einem der Formulare geändert haben, klicken Sie auf den Button **'Übernehmen'** unter dem jeweiligen Formular. Die Anwendung zeigt eine Erfolgsmeldung an. Beachten Sie, dass diese Änderungen erst bei einem **neuen Start** oder **Reset** des Optimierungslaufs wirksam werden, da sie die Initialisierung des Algorithmus beeinflussen.

*   **Auf Defaults zurücksetzen**: Setzt alle Parameter auf die Standardwerte für die aktuell ausgewählte Zielfunktion zurück.
*   **Auf Preset übertragen**: Diese Funktion ist hier nur ein Hinweis; Presets werden auf der 'Presets'-Seite angewendet.

## 5. Algorithmus-Bibliothek ('Algorithmen')

Auf dieser Seite können Sie klassische Optimierungsalgorithmen ausführen und deren Konvergenzmetriken mit HPIO vergleichen.

**5.1 Algorithmus auswählen**
*   **Algorithmus**: Wählen Sie zwischen 'Differential Evolution', 'Particle Swarm Optimization' und 'Genetischer Algorithmus'.
*   **Zielfunktion**: Wählen Sie die Zielfunktion, die der Algorithmus minimieren soll.

**5.2 Algorithmus-Parameter**
Jeder Algorithmus hat spezifische Parameter, die Sie anpassen können:
*   **Seed**: Startwert für den Zufallszahlengenerator.
*   **Iterationen**: Anzahl der Optimierungsschritte.

**Differential Evolution (DE)**
*   **Population**: Anzahl der Individuen in der Population.
*   **Mutation**: Stärke der Mutation.
*   **Crossover**: Wahrscheinlichkeit der Rekombination.

**Particle Swarm Optimization (PSO)**
*   **Schwarmgröße**: Anzahl der Partikel im Schwarm.
*   **Trägheit**: Einfluss der vorherigen Geschwindigkeit auf die aktuelle Bewegung.
*   **Kognitiv**: Einfluss des persönlichen besten Punktes auf die Bewegung.
*   **Sozial**: Einfluss des global besten Punktes auf die Bewegung.

**Genetischer Algorithmus (GA)**
*   **Population**: Anzahl der Individuen in der Population.
*   **Crossover**: Wahrscheinlichkeit der Rekombination.
*   **Mutation**: Wahrscheinlichkeit und Stärke der Mutation.
*   **Tournament-k**: Größe des Turniers für die Selektion von Eltern.

**5.3 Algorithmus starten**
Klicken Sie auf **'Algorithmus starten'**, um den ausgewählten Algorithmus mit den konfigurierten Parametern auszuführen. Die Ergebnisse werden direkt auf der Seite angezeigt.

**5.4 Ergebnisse**
Nach Abschluss der Berechnung werden folgende Informationen angezeigt:
*   **Bestwert**: Der beste gefundene Funktionswert.
*   **Beste Position**: Die Koordinaten des besten gefundenen Punktes.
*   **Liniendiagramm 'best_value'**: Zeigt die Konvergenz des besten Wertes über die Iterationen.
*   **Flächendiagramm 'mean_fitness'**: Zeigt die Entwicklung des durchschnittlichen Fitnesswertes der Population.
*   **Export-Buttons**: Ermöglichen den Download der Metriken als CSV oder JSON.
*   **JSON-Snapshot**: Eine detaillierte JSON-Darstellung der Algorithmus-Konfiguration und der Endergebnisse.

## 6. Presets ('Presets')

Die Seite 'Presets' bietet eine einfache Möglichkeit, vordefinierte oder eigene Konfigurationen zu laden, zu speichern und zu verwalten.

**6.1 Preset wählen**
Wählen Sie aus einer Liste von vordefinierten Presets (z.B. `rastrigin-gpu-pro`, `ackley-gpu-pro`, `himmelblau-cpu-pro`) oder Ihren eigenen geladenen Presets.

**6.2 Preset anwenden**
Klicken Sie auf **'Preset anwenden'**, um die Parameter des ausgewählten Presets auf Ihre aktuelle HPIO-Konfiguration zu übertragen. Dies setzt den `parameter_dirty`-Status und den Controller zurück, sodass die Änderungen beim nächsten Start oder Reset wirksam werden.

**6.3 Preset speichern (JSON)**
Klicken Sie auf **'Preset speichern (JSON)'**, um die *aktuelle* HPIO-Konfiguration als JSON-Datei herunterzuladen. Dies ist nützlich, um Ihre eigenen optimierten Parameterkombinationen zu sichern.

**6.4 Preset laden (JSON)**
Verwenden Sie den **'Preset laden (JSON)'**-Uploader, um eine zuvor gespeicherte JSON-Konfigurationsdatei hochzuladen. Das geladene Preset wird zu Ihrer Liste der 'Custom Presets' hinzugefügt und kann dann ausgewählt und angewendet werden.

**6.5 Diff zur aktuellen Konfiguration**
Dieser Abschnitt zeigt eine Tabelle, die die Unterschiede zwischen Ihrer aktuell geladenen Konfiguration und dem ausgewählten Preset hervorhebt. Dies hilft Ihnen zu verstehen, welche Parameter sich ändern, wenn Sie ein Preset anwenden.

**6.6 Copy as CLI**
Generiert einen Befehlszeilen-Einzeiler, der die aktuelle Konfiguration widerspiegelt und zum Starten des `hpio_record.py`-Skripts verwendet werden kann. Dies ist nützlich für die Automatisierung von Videoaufnahmen oder Batch-Läufen außerhalb der Streamlit-App.

## 7. Aufnahme & Export ('Aufnahme / Export')

Auf dieser Seite können Sie den Optimierungsprozess als Video aufzeichnen und verschiedene Datenartefakte exportieren.

**7.1 Video-Einstellungen**
Konfigurieren Sie die Parameter für die Videoaufnahme:
*   **Dateiname**: Der Name der Ausgabevideodatei.
*   **Format**: Wählen Sie das Videoformat (mp4, mkv, avi).
*   **FPS**: Frames pro Sekunde des Ausgabevideos.
*   **Viz-Frequenz (Frames)**: Legt fest, wie oft (jede n-te Iteration) ein Frame für das Video aufgenommen wird. Unabhängig von der 'Viz-Frequenz' auf der 'Start / Run'-Seite.
*   **Overlay übernehmen**: Ob das Overlay (Iteration / Bestwert) im Video enthalten sein soll.
*   **Encoder-Preset**: Qualitäts- und Geschwindigkeits-Preset für den Video-Encoder (z.B. `medium` für gute Balance, `ultrafast` für schnelle Generierung).
*   **CRF (0-51)**: Constant Rate Factor für die Videoqualität. Niedrigere Werte bedeuten höhere Qualität und größere Dateien, höhere Werte bedeuten geringere Qualität und kleinere Dateien.

Klicken Sie auf **'Speichern'**, um die Video-Parameter zu aktualisieren.

**7.2 Videoaufnahme steuern**
*   **Aufnahme starten**: Beginnt mit dem Sammeln von Frames für das Video. Die Frames werden im Arbeitsspeicher gespeichert (bis zu einem Limit von `MAX_VIDEO_FRAMES`).
*   **Aufnahme stoppen**: Beendet die Frame-Sammlung und versucht, das Video zu speichern. Bei Erfolg wird ein Kurzbericht angezeigt und die Frames aus dem Speicher gelöscht.

Während der Aufnahme wird ein Fortschrittsbalken angezeigt, der die Anzahl der aufgenommenen Frames und die geschätzte Videodauer anzeigt.

**7.3 Artefakte exportieren**
Dieser Abschnitt ermöglicht den Download verschiedener Daten aus dem aktuellen oder letzten Optimierungslauf:
*   **Config exportieren (JSON)**: Lädt die vollständige HPIO-Konfiguration als JSON-Datei herunter.
*   **Best-Trajectory (CSV)**: Exportiert die Historie des besten gefundenen Wertes und seiner Position über die Iterationen als CSV-Datei.
*   **Metriken exportieren (CSV/JSON)**: Lädt detaillierte Metriken pro Iteration (Bestwert, Delta Best, Zeit, FPS) als CSV oder JSON herunter.
*   **Heatmap-Snapshots exportieren (ZIP)**: Erstellt ein ZIP-Archiv mit allen aufgenommenen Heatmap-Frames als PNG-Bilder.
*   **Log exportieren (TXT)**: Speichert den Inhalt der Konsole als Textdatei.

**7.4 Hinweise**
Dieser Abschnitt enthält zusätzliche Informationen zur Videokompatibilität und Encoder-Einstellungen.

## 8. Experimente ('Experimente')

Die Seite 'Experimente' ermöglicht das Durchführen von Batch-Läufen und Parameterstudien, um die Robustheit und Leistung verschiedener Konfigurationen zu bewerten.

Die Experimente sind in drei Tabs organisiert:

**8.1 Seeds-Sweep**
*   **Seeds (Komma, Bereich a-b oder Anzahl n)**: Geben Sie hier eine Liste von Seeds ein, getrennt durch Kommas (z.B. `0, 1, 2, 10`), einen Bereich (z.B. `0-5` für Seeds 0 bis 5) oder eine einzelne Zahl `n` (für Seeds 0 bis n-1). Für jeden Seed wird ein separater Optimierungslauf mit der *aktuellen* HPIO-Konfiguration durchgeführt.
*   **'Seeds-Lauf starten'**: Führt die Batch-Läufe durch und zeigt die Ergebnisse in einer Tabelle an.

**8.2 Preset-Vergleich**
*   **Presets wählen**: Wählen Sie mehrere Presets aus der Liste aus, die miteinander verglichen werden sollen.
*   **Runs pro Preset**: Geben Sie an, wie oft jeder ausgewählte Preset-Lauf wiederholt werden soll. Für jede Wiederholung wird ein neuer, zufälliger Seed verwendet.
*   **'Preset-Benchmark starten'**: Führt die Vergleichsläufe durch und zeigt die Ergebnisse in einer Tabelle an.

**8.3 Parameter-Raster**
Dieser Tab ermöglicht das systematische Testen verschiedener Parameterkombinationen.
*   **Parameter-Key (z.B. field.relax_alpha)**: Geben Sie den vollständigen Pfad zum Parameter an, den Sie variieren möchten (z.B. `field.relax_alpha`, `agent.step`, `w_phase`).
*   **Werte (Komma-Liste)**: Geben Sie eine Komma-getrennte Liste von Werten für den oben angegebenen Parameter ein (z.B. `0.24, 0.26, 0.28`).
*   **Weitere Parameter (key=value;...)**: Hier können Sie zusätzliche Parameter und deren Werte angeben, um komplexere Raster zu erstellen (z.B. `w_phase=0.4;w_intensity=1.0`).
*   **Parameter-CSV (optional)**: Alternativ können Sie eine CSV-Datei hochladen, wobei jede Zeile eine Parameterkombination darstellt. Die Spaltenüberschriften sind die Parameter-Keys.
*   **'Parameter-Raster ausführen'**: Startet die Läufe für alle definierten Parameterkombinationen.

**8.4 Export & Visualisierung**
Nachdem Experimente durchgeführt wurden, können Sie die kombinierten Ergebnisse als JSON-Datei herunterladen.

## 9. Hilfe & Dokumentation ('Hilfe')

Die Seite 'Hilfe' bietet eine Zusammenfassung der wichtigsten Informationen über HPIO, ein Glossar der Parameter, Tipps zur Fehlerbehebung und Performance-Hinweise.

**9.1 Was ist HPIO?**
Eine kurze Einführung in den Hybrid Pheromone Inspired Optimizer und seine Funktionsweise.

**9.2 Parameter-Glossar**
Eine Liste der wichtigsten HPIO-Parameter mit kurzen Erklärungen ihrer Funktion:
*   **relax_alpha**: Stärke der Feldglättung – kleinere Werte = schärfere Peaks.
*   **evap**: Verdunstung, reduziert alte Spuren.
*   **kernel_sigma**: Breite des Gaußfilters in der Relaxation.
*   **step / curiosity / momentum**: Bewegungsparameter der Agenten.
*   **deposit_sigma**: Fußabdruck der Ablage im Grid.
*   **coherence_gain**: Drift in Richtung global best.
*   **w_intensity / w_phase / phase_span_pi**: Gewichtung und Phasen-Spannweite der Ablage.
*   **anneal_***: Lineare Interpolation über die Laufzeit hinweg.
*   **early_patience / early_tol**: Frühabbruch, wenn der Bestwert kaum besser wird.
*   **polish_h**: Parameter für die lokale Nachbesserung.

**9.3 Troubleshooting**
Antworten auf häufige Probleme und Empfehlungen zur Parameteranpassung:
*   **Hängt bei ~1.0 (Rastrigin)**: Erhöhen Sie `w_phase`, verringern Sie `relax_alpha`, `deposit_sigma` und `momentum`.
*   **Ackley bleibt ~0.7**: Verringern Sie `relax_alpha`, erhöhen Sie `w_phase`, verringern Sie `curiosity_to` und `momentum`.
*   **Zappeln**: Erhöhen Sie `momentum` oder `coherence_gain` leicht.

**9.4 Performance-Tipps**
Empfehlungen zur Verbesserung der Anwendungsleistung:
*   Gridgröße ≤ 192² für bessere Performance.
*   Viz-Frequenz hochsetzen, um die Rendering-Last zu reduzieren.
*   Videoaufnahmen getrennt über `hpio_record.py` durchführen.
*   GPU-Beschleunigung aktivieren, falls PyOpenCL verfügbar ist.

**9.5 Kompatibilität & Tools**
Informationen zu benötigten Bibliotheken für den Videoexport (`imageio`, FFmpeg/OpenCV) und zur Generierung von CLI-Befehlen.

