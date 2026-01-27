import time
import board
import adafruit_dht

# Initialisation du capteur sur le GPIO 4
# Le Pi 5 gère les pins différemment, mais la lib 'board' s'en occupe
dht_device = adafruit_dht.DHT11(board.D4)

print("Test du capteur DHT11 (Ctrl+C pour arrêter)")

while True:
    try:
        # Lecture des données
        temperature = dht_device.temperature
        humidity = dht_device.humidity
        
        # Affichage
        if temperature is not None and humidity is not None:
            print(f"Température: {temperature:.1f}°C  |  Humidité: {humidity}%")
        else:
            print("Lecture vide (retry...)")

    except RuntimeError as error:
        # Les erreurs de lecture sont fréquentes sur le DHT11 (checksum error)
        # On affiche l'erreur mais on ne plante pas le programme
        print(f"Erreur de lecture (c'est normal) : {error.args[0]}")
        time.sleep(2.0)
        continue
    
    except Exception as error:
        dht_device.exit()
        raise error

    # Le DHT11 ne peut pas être lu plus d'une fois toutes les 2 secondes
    time.sleep(2.0)