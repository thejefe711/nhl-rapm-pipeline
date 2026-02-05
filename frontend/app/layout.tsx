import type { Metadata } from 'next';
import './globals.css';
import Navbar from '@/components/Navbar';

export const metadata: Metadata = {
    title: 'NHL Analytics | RAPM Player Stats',
    description: 'Advanced hockey analytics with RAPM (Regularized Adjusted Plus-Minus) metrics, player comparisons, and AI-powered insights.',
    keywords: 'NHL, hockey, analytics, RAPM, Corsi, player stats',
    openGraph: {
        title: 'NHL Analytics | RAPM Player Stats',
        description: 'Advanced hockey analytics with RAPM metrics and AI-powered insights.',
        type: 'website',
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
                <main>
                    {children}
                </main>
            </body>
        </html>
    );
}
