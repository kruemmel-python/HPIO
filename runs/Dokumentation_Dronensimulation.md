# 🛸 Dokumentation der Erweiterten Drohnen-Schwarm-Simulation

*(500 Iterationen, periodische Umgebung)*

Diese Dokumentation beschreibt die erweiterten Mechanismen des bio-inspirierten Drohnenschwarm-Systems, das auf der HPIO-Feldarchitektur (Hybrid Pheromone Inspired Optimizer) basiert.
Die vorliegende Version integriert Rollenspezialisierung, Energie- und Heimkehrlogik, prädiktive Kollisionsvermeidung sowie ein kontinuierlich relaxierendes Pheromonfeld Φ.
Das Ergebnis ist ein selbstorganisierendes, resilient arbeitendes Agentensystem mit emergenten, stabilen Mustern.

---

## 1. Systemüberblick

Das System vereint zwei gekoppelte Ebenen:

* **Feld Φ:** Ein kontinuierliches 2D-Gradientenfeld, das durch Drohnenablagen, Evaporation und Relaxation bestimmt wird.
* **Agenten:** Heterogener Schwarm aus 52 Drohnen, die auf Φ reagieren, Spuren ablegen, Energie verbrauchen und autonom zur Basis zurückkehren.

Die Dynamik entsteht aus der Interaktion beider Ebenen:
Drohnen verstärken lokale Maxima im Feld, die wiederum ihre Bewegung lenken – ein klassischer *bio-inspirierter Rückkopplungsmechanismus*.

---

## 2. Detaillierte Parameter-Konfiguration

| **Kategorie**        | **Parameter**      | **Wert**                | **Bedeutung / Wirkung**                                   |                |                      |
| -------------------- | ------------------ | ----------------------- | --------------------------------------------------------- | -------------- | -------------------- |
| **Allgemein**        | Anzahl Drohnen     | **52**                  | Gesamtzahl der aktiven Agenten                            |                |                      |
|                      | Schrittweite       | **3.85**                | Extrem hoch → schnelle Raumabdeckung                      |                |                      |
|                      | Trägheit           | **0.47**                | Mittel-niedrig → hohe Agilität, schnelle Richtungswechsel |                |                      |
|                      | Vermeidungsradius  | **10.50**               | Sehr hoch → großer Sicherheitsabstand, lockeres Cluster   |                |                      |
|                      | Kohäsions-Nachbarn | **11**                  | Lokale Bindung innerhalb kleiner Gruppen                  |                |                      |
|                      | Randbedingungen    | **periodic**            | Feld ohne Ränder → unendlicher Raum                       |                |                      |
| **Rollenverteilung** | Scouts             | **40 % (≈ 21 Drohnen)** | Hohe Neugier, leichte Bewegung, kurze Batterielaufzeit    |                |                      |
|                      | Harvester          | **35 % (≈ 18 Drohnen)** | Geringe Neugier, hohe Ablage, hohe Ausdauer               |                |                      |
|                      | Generalisten       | **25 % (≈ 13 Drohnen)** | Balance zwischen Exploration & Kohäsion                   |                |                      |
| **Batterien**        | Generalist         | **500 Einheiten**       | Basis-Kapazität                                           |                |                      |
|                      | Scout              | **518.68 Einheiten**    | Geringfügig höher für längere Suchflüge                   |                |                      |
|                      | Harvester          | **729.86 Einheiten**    | Sehr hoch → Langzeiteinsatz im Hotspot                    |                |                      |
|                      | Basis (X)          | **(90, 90)**            | Zentral im 160×160-Grid                                   |                |                      |
| **Rollen-Gains**     |                    | **Neugier**             | **Kohärenz**                                              | **Vermeidung** | **Σ (Ablage-Sigma)** |
| Generalist           | 1.67               | 0.36                    | 0.80                                                      | 1.68           |                      |
| Scout                | 1.24               | 0.36                    | 0.65                                                      | 0.88           |                      |
| Harvester            | 0.15               | 0.25                    | 0.85                                                      | 2.99           |                      |
| **Feld-Parameter Φ** | Verdunstung        | **0.03**                | Langsame Zerfallsrate → langlebige Spuren                 |                |                      |
|                      | Relaxation         | **0.40**                | Mittelhoch → schnelles Glätten, Cluster-Verschmelzung     |                |                      |

---

## 3. Schlüssel-Dynamiken des Systems

### 3.1 Explorations-Turbo

Die Kombination aus sehr hoher Schrittweite (3.85) und ausgeprägter Neugier bei Scouts (1.24) und Generalisten (1.67) erzeugt eine intensive **Explorationsphase**.
Das Feld Φ wird in kürzester Zeit gleichmäßig abgetastet – getrieben durch schnelle, stochastisch modulierte Bewegungen.

### 3.2 Verstärkungs-Spezialisierung

Harvester mit sehr hoher Ablage-Sigma (2.99) und minimaler Neugier (0.15) bleiben nahezu stationär in gefundenen Maxima.
Sie erzeugen breite, stabile Ablage-Zonen, die sich durch Relaxation (0.40) zu dominanten Hotspots vereinigen.

### 3.3 Harter Abstands-Druck

Der Vermeidungsradius (10.50) verhindert Clusterkollapse.
Drohnen ordnen sich in einem **lockeren, nahezu ringförmigen Muster**, wodurch die Gesamtstruktur erhalten bleibt und interne Bewegung möglich bleibt.

### 3.4 Ressourcen-Rotation

Das Batteriemanagement sorgt für **kontinuierlichen Agentenfluss** zwischen Basis und Hotspot:
Scouts & Generalisten kehren früher zurück, Harvester halten den Hotspot aktiv.
So bleibt der Schwarm immer vollzählig (52 / 52 aktiv).

---

## 4. Phasenanalyse des Simulationsverlaufs

### **Phase 1 – Aggressive Initial-Exploration (Iterationen 5–100)**

* **Verhalten:** Rasante Expansion in alle Richtungen; zahlreiche kleine Φ-Maxima entstehen.
* **Treiber:** Große Schrittweite + hohe Neugier.
* **Feld:** Viele diffuse Spuren; Relaxation verschmilzt sie zu einem großflächigen Suchgebiet.
* **Ränder:** Durch *periodic wrapping* keine Blockade – Raum erscheint unendlich.
* **Energie:** Alle 52 Drohnen aktiv; Batterieverbrauch steigt schnell an.

---

### **Phase 2 – Cluster-Stabilisierung & Beginnende Rotation (Iterationen 100–300)**

* **Verhalten:**

  * Bildung eines zentralen Groß-Clusters um das dominante Φ-Maximum.
  * Harvester übernehmen das Zentrum; Scouts & Generalisten umkreisen.
* **Rückkehrlogik:**

  * Erste Energie-Rückkehr ab Iteration ≈ 245.
  * Scouts / Generalisten aktivieren *returning_to_base*, stoppen Ablage, folgen Heimvektor.
  * Im Video als **weiße Quadrate** sichtbar, die zur Basis zurückfliegen.
* **Effekt:**

  * Harvester bleiben im Feld (hohe Kapazität).
  * Hotspot bleibt permanent aktiv – nahtlose Arbeitsteilung entsteht.

---

### **Phase 3 – Stabiler Missionsbetrieb (Iterationen 300–495)**

* **Verhalten:**

  * Stationäres Gleichgewicht zwischen Such-, Ablage- und Rückkehr-Drohnen.
  * Das Cluster oszilliert leicht, bleibt aber stabil.
* **Mechanismus:**

  * **Adaptive Vermeidung + Gradienten-Folgen** erzeugen einen “tanzenden” Gleichgewichtszustand.
  * Zufällige Bewegungsanteile halten die Struktur lebendig.
* **Ressourcenfluss:**

  * Drohnen kehren kontinuierlich, asynchron zurück; Ladevorgänge an der Basis gleichen Energiezyklen aus.
  * Aktive Anzahl konstant: 52 / 52.

---

## 5. Zusammenfassung der Emergenz

| **Prinzip**                    | **Beobachtung / Resultat**                                                            |
| ------------------------------ | ------------------------------------------------------------------------------------- |
| **Exploration ↔ Exploitation** | Scouts & Generalisten suchen, Harvester verstärken Φ-Maxima.                          |
| **Rollenspezialisierung**      | Unterschiedliche Neugier-, Sigma- und Energieparameter erzeugen echte Arbeitsteilung. |
| **Räumliche Organisation**     | Hoher Vermeidungsradius → ringförmige, stabile Großcluster.                           |
| **Ressourcen-Resilienz**       | Durch asynchrone Rückkehr bleibt die Aktivität konstant – kein Totalausfall.          |
| **Periodische Welt**           | Keine Randartefakte, vollflächige Nutzung des Suchraums.                              |

---

## 6. Schlussfolgerung

Die gewählte Konfiguration bildet ein **autonomes, langlebiges Schwarm-Ökosystem**:

* **Effizient:** Maximale Flächenabdeckung durch aggressive Explorationsphase.
* **Stabil:** Harvester-Ankerpunkte verhindern Drift und Verlust des globalen Maximums.
* **Resilient:** Energiezyklen laufen kontinuierlich – kein Shutdown durch gleichzeitige Entladung.
* **Organisch:** Adaptive Bewegungslogik, variable Rolleninteraktion und periodische Umgebung erzeugen natürlich wirkende, emergente Dynamik.

Das System stellt eine gelungene **Fusion aus Optimierungslogik und Schwarmphysik** dar – ein künstliches Ökosystem, das sich selbst organisiert, reguliert und erhält.

---

*(Erstellt auf Basis der erweiterten HPIO-Drohnenarchitektur, 495 – 500 Iterationen, Analyse Ralf Krümmel 2025)*
