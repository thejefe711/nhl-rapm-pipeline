import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import CommandPalette from '@/components/CommandPalette';

export const metadata: Metadata = {
    title: 'Data-Puck.com | Advanced RAPM Player Stats & Hockey Metrics',
    description: 'The most comprehensive NHL analytics platform. RAPM, Expected Goals, High Danger xG, Playmaking, and AI-powered player insights across 6+ seasons and 800+ players.',
    keywords: 'Data-Puck, NHL, hockey, analytics, RAPM, Corsi, xG, expected goals, player stats, advanced stats, hockey analytics',
    openGraph: {
        title: 'Data-Puck.com | Advanced RAPM Player Stats',
        description: 'Go beyond box scores. Explore RAPM metrics, expected goals models, and AI-powered insights for 800+ NHL players.',
        type: 'website',
        siteName: 'Data-Puck.com',
    },
    twitter: {
        card: 'summary_large_image',
        title: 'Data-Puck.com | RAPM Player Stats',
        description: 'Advanced hockey analytics with RAPM, xG, and AI-powered player insights.',
    },
    robots: {
        index: true,
        follow: true,
    },
};

export default function RootLayout({
    children,
}: {
    children: React.ReactNode;
}) {
    return (
        <html lang="en">
            <head>
                <link rel="preconnect" href="https://fonts.googleapis.com" />
                <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
            </head>
            <body>
                <Navbar />
                <CommandPalette />
                <main>
                    {children}
                </main>
                <Footer />
            </body>
        </html>
    );
}
