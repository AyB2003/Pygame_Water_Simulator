

import pygame
import cv2
import numpy as np
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import random
from V1 import *

scale = 0.02 # each pixel is 0.2 mm or 2 µm or 0.02 cm

pygame.init()
screen = pygame.display.set_mode((200, 500))
H = screen.get_height()
W = screen.get_width()
print(H,W)
fourcc = cv2.VideoWriter_fourcc(*'H264')
fps = 60
duration = 10
total_frames = duration * fps
seconds = 10
dt = 0.01
Radius = 10
smoothing_radius = Radius*2
previous_completion = 0
video_writer = cv2.VideoWriter('output.mp4', fourcc, fps, (W, H))
slider = Slider(screen, 10, 50, 70, 10, min=5, max=20, step=1)
output = TextBox(screen, 90, 40, 30, 30, fontSize=20)
Density = TextBox(screen, 800, 40, 100, 30, fontSize = 20)
output.disable()
slider.value = 5
clock = pygame.time.Clock()
running = True
smallfont = pygame.font.SysFont('Corbel', 40)
smallerfont = pygame.font.SysFont('Corbel', 15)
text1 = smallfont.render('+', True, (255,255,255))
text2 = smallfont.render('-', True, (255,255,255))
text3 = smallerfont.render('G_ON', True, (255, 255, 255))
text4 = smallerfont.render('G_OFF', True, (255, 255, 255))


particles = []
frame_count = 0
particleline = 1
P_numbers = particleline
selector = 0
d_pressed = False
space_pressed = False
mouse_pressed = False
gravity = False
i = 0
selected = False
selectedparticale = 0
rect = (0,0, W, H)

for i in range(particleline):
    while True:
        a = random.randint(-1,1)
        b = random.randint(-1,1)
        if a != 0 and b != 0:
            break
    particles.append(Particle(Radius, 2000, (a,0), (0,0), x_position= 100 ,y_position=random.randint(W/2 - 100,W/2 + 100),color=(41,220,214)))

while running:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if  0 <= mouse[0] <= 60 and 0 <= mouse[1] <= 40:
                if not mouse_pressed:
                    P_numbers += 1
                    while True:
                        a = random.randint(-4,4)
                        b = random.randint(-4,4)
                        if a != 0 and b != 0:
                            break
                    particles.append(Particle(
                        diameter=Radius,
                        mass=2000,
                        speed=(a, b),
                        acceleration=(0, 13.625),
                        x_position=screen.get_width()/1.5,
                        y_position=screen.get_height()/1.5,
                        color=(41,220,214)
                    ))
                    mouse_pressed = True
                    print(f'there are {len(particles)} particales')
            elif 60 <= mouse[0] <= 120 and 0 <= mouse[1] <= 40:
                if not mouse_pressed and P_numbers > 0:
                    P_numbers -= 1
                    particles.pop()
            elif 120 <= mouse[0] <= 180 and 0 <= mouse[1] <= 40:
                if not mouse_pressed:
                    gravity = False
            elif 180 <= mouse[0] <= 240 and 0 <= mouse[1] <= 40:
                if not mouse_pressed:
                    gravity = True
        elif event.type == pygame.MOUSEBUTTONUP:
            mouse_pressed = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_d:
                if not d_pressed:
                    if selectedparticale <= P_numbers:
                        particles[selectedparticale].P_stat(particles)
                    #print(n)
                    #Obj.O_stat()
                    d_pressed = True
            elif event.key == pygame.K_SPACE:
                if not space_pressed:
                    selected = True
                    selectedparticale = random.randint(0,P_numbers)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_d:
                d_pressed = False
            elif event.key == pygame.K_SPACE:
                space_pressed = False
    mouse = pygame.mouse.get_pos()
    '''Obj = Object(300,300,False,screen.get_width()/2 - 300/2,screen.get_height()/2 - 300/2)
    Controls = Object(240,40,False,0,0)
    output.setText(slider.getValue())
    pygame.draw.rect(screen, (255, 255, 0), [Obj.position.x,Obj.position.y, Obj.length,Obj.width], 2)
    pygame.draw.rect(screen, (50,50,50), [0,0,60,40])
    pygame.draw.rect(screen, (50,50,50), [60,0,60,40])
    pygame.draw.rect(screen, (50,50,50), [120,0,60,40])
    pygame.draw.rect(screen, (50,50,50), [180,0,60,40])'''

    '''if   0 <= mouse[0] <= 60 and 0 <= mouse[1] <= 40:
        pygame.draw.rect(screen, (100,100,100), [0,0,60,40])
    if 60 <= mouse[0] <= 120 and 0 <= mouse[1] <= 40:
        pygame.draw.rect(screen, (100,100,100), [60,0,60,40])
    if 120 <= mouse[0] <= 180 and 0 <= mouse[1] <= 40:
        pygame.draw.rect(screen, (100,100,100), [120,0,60,40])
    if 180 <= mouse[0] <= 240 and 0 <= mouse[1] <= 40:
        pygame.draw.rect(screen, (100,100,100), [180,0,60,40])
    pygame.draw.rect(screen, (255,255,255), [0,0,240,40], 2)'''
    screen.fill("black")
    for p in particles:
        if gravity:
            p.acceleration = pygame.Vector2(0,13.26)
        else:
            p.acceleration = pygame.Vector2(0,0)

    #screen.blit(text1, (20,0))
    #screen.blit(text2, (80,0))
    #screen.blit(text3, (130,10))
    #screen.blit(text4, (190,10))
    #Obj.objectcollision(particles)
    #Controls.objectcollision(particles)
    #n = collision(particles)
    #density_heat_map(particles)
    #modified_heat_map(particles, smoothing_radius)
    #detection_radius(screen, particles, smoothing_radius)
    #densityequalizer(particles, move_distance=1, clump_threshold=3, neighbor_distance=1.0)
    Pressure_Force(particles,smoothing_radius)
    for particle in particles:
        particle.movement(screen, dt)
        particle.draw(screen)
    #apply_gaussian_filter_to_region(screen, sigma=10, rect=rect)
    #pygame_widgets.update(events)
    frame = pygame.surfarray.array3d(screen)
    frame = np.transpose(frame, (1, 0, 2))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)
    pygame.display.flip()
    frame_count += 1
    if frame_count == total_frames:
        running = False
    completion_percentage = round((frame_count/total_frames*100),1)
    if abs(previous_completion - completion_percentage) >= 1:
        print(f'The file is {completion_percentage}% finished')
        print(f'The particle densities are {Particledensities(particles, smoothing_radius)[0:5]}')
        print(f'The particle pressures are {Particlepressures(particles, smoothing_radius)[0:5]}')
        F = Forcecalculator(particles, smoothing_radius)[0:5]
        if F:
            print(f'The forces are {Forcecalculator(particles, smoothing_radius)[0:5]}')
        previous_completion = completion_percentage
    if frame_count % 60 == 0:
        seconds += 1
        print(f'The seconds are {seconds}')


    clock.tick(60)

video_writer.release()
pygame.quit()
