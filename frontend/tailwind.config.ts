
import type { Config } from "tailwindcss";

export default {
	darkMode: ["class"],
	content: [
		"./pages/**/*.{ts,tsx}",
		"./components/**/*.{ts,tsx}",
		"./app/**/*.{ts,tsx}",
		"./src/**/*.{ts,tsx}",
	],
	prefix: "",
	theme: {
		container: {
			center: true,
			padding: '2rem',
			screens: {
				'2xl': '1400px'
			}
		},
		extend: {
			colors: {
				border: 'hsl(var(--border))',
				input: 'hsl(var(--input))',
				ring: 'hsl(var(--ring))',
				background: 'hsl(var(--background))',
				foreground: 'hsl(var(--foreground))',
				primary: {
					DEFAULT: 'hsl(var(--primary))',
					foreground: 'hsl(var(--primary-foreground))'
				},
				secondary: {
					DEFAULT: 'hsl(var(--secondary))',
					foreground: 'hsl(var(--secondary-foreground))'
				},
				destructive: {
					DEFAULT: 'hsl(var(--destructive))',
					foreground: 'hsl(var(--destructive-foreground))'
				},
				muted: {
					DEFAULT: 'hsl(var(--muted))',
					foreground: 'hsl(var(--muted-foreground))'
				},
				accent: {
					DEFAULT: 'hsl(var(--accent))',
					foreground: 'hsl(var(--accent-foreground))'
				},
				popover: {
					DEFAULT: 'hsl(var(--popover))',
					foreground: 'hsl(var(--popover-foreground))'
				},
				card: {
					DEFAULT: 'hsl(var(--card))',
					foreground: 'hsl(var(--card-foreground))'
				},
				// The Office custom colors
				dundies: {
					DEFAULT: "#FFD700", // Dundies gold
					foreground: "#000000"
				},
				schrute: {
					DEFAULT: "#4CAF50", // Schrute Farms green
					foreground: "#FFFFFF"
				},
				chilis: {
					DEFAULT: "#D32F2F", // Chili's red
					foreground: "#FFFFFF"
				},
				pretzel: {
					DEFAULT: "#F57C00", // Pretzel Day orange
					foreground: "#FFFFFF"
				},
				officeBackground: {
					DEFAULT: "#FAF3E0", // Off-white background
					foreground: "#000000"
				},
				sidebar: {
					DEFAULT: 'hsl(var(--sidebar-background))',
					foreground: 'hsl(var(--sidebar-foreground))',
					primary: 'hsl(var(--sidebar-primary))',
					'primary-foreground': 'hsl(var(--sidebar-primary-foreground))',
					accent: 'hsl(var(--sidebar-accent))',
					'accent-foreground': 'hsl(var(--sidebar-accent-foreground))',
					border: 'hsl(var(--sidebar-border))',
					ring: 'hsl(var(--sidebar-ring))'
				}
			},
			fontFamily: {
				pixel: ['"VT323"', 'monospace'],
				pixelify: ['"Pixelify Sans"', 'cursive'],
				bangers: ['"Bangers"', 'cursive'],
				marker: ['"Permanent Marker"', 'cursive']
			},
			borderRadius: {
				lg: 'var(--radius)',
				md: 'calc(var(--radius) - 2px)',
				sm: 'calc(var(--radius) - 4px)'
			},
			keyframes: {
				'accordion-down': {
					from: {
						height: '0'
					},
					to: {
						height: 'var(--radix-accordion-content-height)'
					}
				},
				'accordion-up': {
					from: {
						height: 'var(--radix-accordion-content-height)'
					},
					to: {
						height: '0'
					}
				},
				'pulse-pixel': {
					'0%, 100%': { 
						transform: 'scale(1)' 
					},
					'50%': { 
						transform: 'scale(1.05)' 
					}
				},
				'blink': {
					'0%, 100%': { 
						opacity: '1'
					},
					'50%': { 
						opacity: '0.5' 
					}
				},
				'typing-dots': {
					'0%': { content: '"."' },
					'33%': { content: '".."' },
					'66%': { content: '"..."' },
					'100%': { content: '"."' }
				},
				'slide-up': {
					'0%': { 
						transform: 'translateY(100%)',
						opacity: '0'
					},
					'100%': { 
						transform: 'translateY(0)',
						opacity: '1'
					}
				},
				'achievement-pop': {
					'0%': {
						transform: 'scale(0.5)',
						opacity: '0'
					},
					'10%': {
						transform: 'scale(1.1)',
						opacity: '1'
					},
					'20%': {
						transform: 'scale(1)'
					},
					'80%': {
						transform: 'scale(1)',
						opacity: '1'
					},
					'100%': {
						transform: 'scale(0.9)',
						opacity: '0'
					}
				},
				'pixel-shine': {
					'0%': {
						backgroundPosition: '0% 0%'
					},
					'100%': {
						backgroundPosition: '200% 0%'
					}
				}
			},
			animation: {
				'accordion-down': 'accordion-down 0.2s ease-out',
				'accordion-up': 'accordion-up 0.2s ease-out',
				'pulse-pixel': 'pulse-pixel 2s infinite ease-in-out',
				'blink': 'blink 1.5s infinite',
				'typing': 'typing-dots 1.5s infinite steps(3)',
				'slide-up': 'slide-up 0.3s ease-out forwards',
				'achievement-pop': 'achievement-pop 4s forwards',
				'pixel-shine': 'pixel-shine 3s linear infinite'
			},
			backgroundImage: {
				'paper-texture': "url('https://www.transparenttextures.com/patterns/exclusive-paper.png')",
				'pixel-grid': "url('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAIklEQVQYV2NkQAIzZ878zwgi0IXgAsiCYAGyIFwAXRBZAAAtNQgFvWj5MgAAAABJRU5ErkJggg==')"
			}
		}
	},
	plugins: [require("tailwindcss-animate")],
} satisfies Config;
