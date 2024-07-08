import pygame
import os
import math
import sys
import neat

SCREEN_WIDTH = 1244
SCREEN_HEIGHT = 1016
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "track.png"))


class Car(pygame.sprite.Sprite): #inherit from sprite class in pygame that recognise objects which is the car in our case
    def __init__(self): #initialization of attributes
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(490, 820)) #image rectangle
        self.vel_vector = pygame.math.Vector2(0.8, 0) #velocity victor
        self.angle = 0
        self.rotation_vel = 5
        self.direction = 0 #-1 whenever we turn left and 1 when right otherwise zero
        self.alive = True #cuz its on the street when the game starts
        self.radars = [] #list of all the data collected by the individual radars

    def update(self): #update the car
        self.radars.clear() #clear the list each time this function is called
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60): #for loop for 5 diff radars
            self.radar(radar_angle)
        self.collision()
        self.data()

    def drive(self):
        self.rect.center += self.vel_vector * 6 #increment rectangle by velocity vector

    def collision(self): #if it goes off the street
        length = 40 #distance between the center of the car and the collision point
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),#two collission points which is left and right
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Die on Collision
        #color of either collision point is green then the car in grass and it will know that the car died
        if SCREEN.get_at(collision_point_right) == pygame.Color(2, 105, 31, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(2, 105, 31, 255):
            self.alive = False

        # Draw Collision Points
        # #drawing the two blue collision points in da front of the car
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self): #scale down the original image to smaller image
        if self.direction == 1: #rotate right
            self.angle -= self.rotation_vel
            self.vel_vector.rotate_ip(self.rotation_vel)
        if self.direction == -1: #rotate left
            self.angle += self.rotation_vel
            self.vel_vector.rotate_ip(-self.rotation_vel)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1) #scale down the image
        self.rect = self.image.get_rect(center=self.rect.center) #get the rectangle of the car image

    def radar(self, radar_angle):
        length = 0 #length of the radar
        x = int(self.rect.center[0]) #x and y of the center of the car
        y = int(self.rect.center[1])

        while not SCREEN.get_at((x, y)) == pygame.Color(2, 105, 31, 255) and length < 200: #extending rader until it hits the grassy area
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw Radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3) #dot on the top of the radar

        dist = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                             + math.pow(self.rect.center[1] - y, 2))) #function that stores the distination between the center of the car and the tip of the radar

        self.radars.append([radar_angle, dist]) #add the data collected to the radar list

    def data(self):
        input = [0, 0, 0, 0, 0] #list that has 5 zeros
        for i, radar in enumerate(self.radars):           #fill input list with data from rader and then we return the input list
            input[i] = int(radar[1])
        return input


def remove(index): #remove the car that run into the grass area it takes index cuz every car has its own index
    cars.pop(index)
    ge.pop(index)
    nets.pop(index)


def eval_genomes(genomes, config): #represented by the individual cars, this function is responsiple for giving each car a fitness score which is how well the car drives the track and its being determind by how long the car stays alive
    global cars, ge, nets

    cars = []
    ge = []
    nets = []

    for genome_id, genome in genomes:
        cars.append(pygame.sprite.GroupSingle(Car())) #cars are filled with genome
        ge.append(genome)
        net = neat.nn.FeedForwardNetwork.create(genome, config) #net list we add the net
        nets.append(net)
        genome.fitness = 0

    run = True
    while run: #standard code for quiting the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0)) #display track on display with pygame language

        if len(cars) == 0:
            break

        for i, car in enumerate(cars): #put fitness to each car
            ge[i].fitness += 1
            if not car.sprite.alive: #remove cars that touch the grass
                remove(i)

        for i, car in enumerate(cars): #how each individual car drive which is determined by
            output = nets[i].activate(car.sprite.data())# driving will be determined by te output of the nueral network and the output is generated by activation function which take the rader data
            if output[0] > 0.7:
                car.sprite.direction = 1 #right
            if output[1] > 0.7:
                car.sprite.direction = -1 #left
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0 #keep going

        # Update
        #we must update all the cars
        for car in cars:
            car.draw(SCREEN)# draw car on screen window
            car.update()
        pygame.display.update()


# Setup NEAT Neural Network
def run(config_path):
    global pop
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True)) #statistics reporter provide information about every car generation
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    pop.run(eval_genomes, 50) #run the population


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)